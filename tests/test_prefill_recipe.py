"""Qualified prompt-kernel defaults, explicit fallback and unchanged safety. CPU only."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PrefillRecipe(unittest.TestCase):
    def test_switch_default_on_and_explicit_environment_fallback(self):
        serve = load('serve')
        for enabled in ['0', '1']:
            with patch.dict(os.environ, {'QWEN_PREFILL_KERNEL': enabled}), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(serve.main(['--model','/model','--vision-weights','/tower','--print-config']),0)
            config = json.loads(out.getvalue())
            self.assertEqual(config['prompt_expert_kernels'], enabled == '1')
            self.assertEqual(config['engine']['max_len'],262144)
            self.assertEqual(config['engine']['streams'],2)
            self.assertEqual(config['engine']['depth'],6)
            self.assertEqual(config['engine']['confidence'],0.70)
            self.assertEqual(config['engine']['kv_dtype'],'bf16')
            self.assertTrue(config['thinking_default'])
            self.assertEqual(config['vision_max_images'],64)
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(serve.prefill_kernel_enabled())
        with patch.dict(os.environ, {'QWEN_PREFILL_KERNEL':'yes'}):
            with self.assertRaises(ValueError):serve.prefill_kernel_enabled()

    def test_serve_explicit_flags_override_host_environment(self):
        mod = load('serve')
        for flag, expected in [('--prefill-kernels', True), ('--legacy-prefill-kernels', False)]:
            with patch.dict(os.environ, {'QWEN_PREFILL_KERNEL': str(int(not expected))}), contextlib.redirect_stdout(io.StringIO()) as out:
                mod.main(['--model','/model','--vision-weights','/tower',flag,'--print-config'])
            self.assertIs(json.loads(out.getvalue())['prompt_expert_kernels'], expected)

    def test_serve_sets_actual_runtime_switch_before_engine_construction(self):
        mod = load('serve')
        seen = []
        engine_module = ModuleType('tensorfold.families.qwen4_exp.cuda.engine')
        server_module = ModuleType('tensorfold.cuda.server')
        def engine(*args, **kwargs):
            seen.append(os.environ.get('QWEN_PREFILL_KERNEL'))
            return SimpleNamespace(close=lambda: None)
        setattr(engine_module, 'FlashNextEngine', engine)
        setattr(server_module, 'App', lambda *args, **kwargs: object())
        setattr(server_module, 'make_handler', lambda app: object())
        setattr(server_module, 'Server', lambda *args: SimpleNamespace(serve_forever=lambda: None, server_close=lambda: None))
        modules = {engine_module.__name__: engine_module, server_module.__name__: server_module}
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model/'config.json').write_text(json.dumps({'model_type':'qwen4_exp','quantization_config':{'quant_method':'exl3'}}))
            tower = model/'tower'; tower.touch()
            for flags, expected in [([], '1'), (['--prefill-kernels'], '1'), (['--legacy-prefill-kernels'], '0')]:
                with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, modules), patch.object(mod.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(mod.main(['--model',str(model),'--vision-weights',str(tower),*flags]),0)
                self.assertEqual(seen[-1], expected)

    def test_launcher_does_not_inherit_an_accidental_host_switch(self):
        mod = load('supervise')
        args = [Path('/model'),Path('/vision/tower'),Path('/cache'),'owned',8001,Path('/cache/cid')]
        with patch.dict(os.environ, {'QWEN_PREFILL_KERNEL':'0'}):
            default=mod.docker_command(*args)
            enabled=mod.docker_command(*args,prefill_kernels=True)
            legacy=mod.docker_command(*args,prefill_kernels=False)
        self.assertEqual(default, enabled)
        self.assertIn('QWEN_PREFILL_KERNEL=1',default)
        self.assertIn('QWEN_PREFILL_KERNEL=0',legacy)
        self.assertEqual([x for x in default if not x.startswith('QWEN_PREFILL_KERNEL=')], [x for x in legacy if not x.startswith('QWEN_PREFILL_KERNEL=')])
        self.assertIn('--memory-swap',default)
        self.assertNotIn('QWEN_PREFILL_READAHEAD=1',default)

    def test_dry_run_forwards_defaults_compatibility_and_fallback(self):
        mod=load('supervise')
        for flags, switch in [([], '1'), (['--prefill-kernels'], '1'), (['--legacy-prefill-kernels'], '0')]:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(mod.main(['/model','/tower',*flags,'--dry-run']),0)
            self.assertIn('QWEN_PREFILL_KERNEL='+switch,json.loads(out.getvalue()))

    def test_installed_service_preserves_both_modes_explicitly(self):
        mod=load('install_service')
        self.assertIn('"--prefill-kernels"',mod.render('/model','/tower'))
        self.assertEqual(mod.render('/model','/tower'),mod.render('/model','/tower',prefill_kernels=True))
        text=mod.render('/model','/tower',prefill_kernels=False)
        self.assertIn('"--legacy-prefill-kernels"',text)
        self.assertNotIn('"--prefill-kernels"',text)
        self.assertIn('MemorySwapMax=0',text)
        self.assertIn('Restart=no',text)

    def test_installer_shell_wrapper_forwards_both_flags_without_installing(self):
        for flags, selected in [([], '--prefill-kernels'), (['--prefill-kernels'], '--prefill-kernels'), (['--legacy-prefill-kernels'], '--legacy-prefill-kernels')]:
            proc=subprocess.run(['bash',str(ROOT/'scripts/install_service.sh'),'/model','/vision/vision_tower_bf16.safetensors',*flags,'--dry-run'],capture_output=True,text=True,check=True)
            self.assertIn('"'+selected+'"',proc.stdout)
            self.assertNotIn('Installed ',proc.stdout)

    def test_flags_are_mutually_exclusive_everywhere(self):
        for name, args in [('serve',['--model','/model','--vision-weights','/tower','--print-config']),('supervise',['/model','/tower','--dry-run']),('install_service',['/model','/vision/vision_tower_bf16.safetensors','--dry-run'])]:
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    load(name).main([*args,'--prefill-kernels','--legacy-prefill-kernels'])
                self.assertEqual(caught.exception.code,2)

    def test_capacity_extra_is_accounted_without_context_or_policy_change(self):
        mod=load('supervise')
        self.assertEqual(mod.PROMPT_WORKSPACE_EXTRA,480285400)
        self.assertEqual(mod.minimum_preload_available(),96700886352)
        self.assertEqual(mod.minimum_preload_available(False),96220600952)
        self.assertEqual(mod.minimum_preload_available()-mod.minimum_preload_available(False),480285400)
        boundary={'mem_available':7*mod.GIB,'swap_used':mod.GIB,'disk_free':20*mod.GIB}
        self.assertIsNone(mod.breach(boundary,{'swap_used':0}))
        self.assertIsNotNone(mod.breach({**boundary,'mem_available':7*mod.GIB-1},{'swap_used':0}))
        self.assertIsNotNone(mod.breach({**boundary,'swap_used':mod.GIB+1},{'swap_used':0}))
        self.assertIsNotNone(mod.breach(boundary,{'swap_used':0},container_swap=1))

    def test_actual_admission_boundary_defaults_and_fallback(self):
        mod=load('supervise')
        for flags, minimum in [([],96700886352),(['--prefill-kernels'],96700886352),(['--legacy-prefill-kernels'],96220600952)]:
            for available, message in [(minimum-1,'Insufficient preload headroom'),(minimum,'Recipe container already exists')]:
                with self.subTest(flags=flags,available=available), patch.object(mod,'validate_artifacts'), patch.object(mod,'sample',return_value={'mem_available':available,'swap_used':0,'disk_free':30*mod.GIB}), patch.object(mod.subprocess,'run',return_value=SimpleNamespace(stdout='existing\n')) as run, patch.object(mod.subprocess,'Popen',side_effect=AssertionError('no container launch')):
                    with self.assertRaisesRegex(SystemExit,message):
                        mod.main(['/model','/tower',*flags])
                    self.assertEqual(run.call_count,int(available==minimum))

    def test_image_patch_requirement_default_on_and_legacy_fallback(self):
        mod=load('supervise')
        labels={'qwen.runtime.revision':'609ca419abecebdc5a059498a613680bd3aa847f'}
        image={'Config':{'Labels':labels}}
        with self.assertRaisesRegex(SystemExit,'Prompt-kernel image patch mismatch'):
            mod.validate_image(image)
        mod.validate_image(image,prefill_kernels=False)
        labels['qwen.prompt-expert.patch.sha256']='wrong'
        with self.assertRaisesRegex(SystemExit,'Prompt-kernel image patch mismatch'):
            mod.validate_image(image)
        labels['qwen.prompt-expert.patch.sha256']=hashlib.sha256((ROOT/'patches/prompt-expert-kernels.patch').read_bytes()).hexdigest()
        mod.validate_image(image)
        for enabled in [False,True]:
            with self.assertRaisesRegex(SystemExit,'Runtime image pin mismatch'):
                mod.validate_image({'Config':{'Labels':{}}},prefill_kernels=enabled)

    def test_main_enforces_image_requirement_before_creating_cache_or_container(self):
        mod=load('supervise')
        image={'Config':{'Labels':{'qwen.runtime.revision':'609ca419abecebdc5a059498a613680bd3aa847f'}}}
        with patch.object(mod,'validate_artifacts'), patch.object(mod,'sample',return_value={'mem_available':100*mod.GIB,'swap_used':0,'disk_free':30*mod.GIB}), patch.object(mod.subprocess,'run',return_value=SimpleNamespace(stdout='')), patch.object(mod.subprocess,'check_output',return_value=json.dumps([image])), patch.object(mod.Path,'mkdir',side_effect=AssertionError('no cache mutation')), patch.object(mod.subprocess,'Popen',side_effect=AssertionError('no container launch')):
            with self.assertRaisesRegex(SystemExit,'Prompt-kernel image patch mismatch'):
                mod.main(['/model','/tower'])

    def test_hashed_patch_is_installed_after_vision_patch(self):
        p=ROOT/'patches/prompt-expert-kernels.patch'
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        docker=(ROOT/'docker/Dockerfile').read_text()
        self.assertIn(sha,docker)
        self.assertIn(sha,(ROOT/'scripts/supervise.py').read_text())
        install='pip install --no-deps --no-build-isolation /opt/TensorFold'
        self.assertLess(docker.index('apply /opt/recipe/patches/vision-prefix-cache.patch'),docker.index('apply /opt/recipe/patches/prompt-expert-kernels.patch'))
        self.assertLess(docker.index('apply /opt/recipe/patches/prompt-expert-kernels.patch'),docker.index(install))
        source=p.read_text()
        self.assertIn('legacy212/*.cu',source)
        self.assertIn('legacy212/experts_grouped.cuh',source)
        self.assertNotIn('ngram_ahead.py',source)
        self.assertIn('prompt=b.prefill',source)
        self.assertIn('admission_extra',source)

    def test_public_patch_verifier_covers_legacy_and_prompt_sources(self):
        mod=load('verify_prefill_patch')
        self.assertEqual(len(mod.EXPECTED),18)
        self.assertEqual(len([x for x in mod.EXPECTED if '/legacy212/' in x]),6)
        self.assertIn('src/tensorfold/cuda/exl3/experts_prompt.cuh',mod.EXPECTED)
        self.assertIn('tests/cuda/test_exl3_prompt_experts.py',mod.EXPECTED)
        self.assertTrue(all(len(x)==64 for x in mod.EXPECTED.values()))

    def test_manifest_descriptive_defaults_do_not_change_downloaded_weight_pin(self):
        manifest=json.loads((ROOT/'manifests/exl3-405.json').read_text())
        defaults=manifest.pop('serving_defaults')
        self.assertIs(defaults['prompt_expert_kernels'],True)
        self.assertEqual(defaults['legacy_fallback_flag'],'--legacy-prefill-kernels')
        self.assertEqual(hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'c187b5f2886f74f4151286ea351489ea37eb0fd8118b5a8c33bcdf3b38f5b2b8')


if __name__=='__main__':unittest.main()
