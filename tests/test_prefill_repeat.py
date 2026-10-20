"""Offline contracts for the measured three-repeat prompt-kernel publication."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[1]

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

class PrefillRepeat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/'results/prefill-repeat.json').read_text())
        cls.arms=list(cls.data['stages'].values())

    def test_scope_is_only_the_qualified_prompt_comparison(self):
        self.assertEqual(self.data['schema'],'qwen-prefill-repeat-v1')
        self.assertEqual(self.data['decision']['verdict'],'PROMOTE')
        self.assertEqual(self.data['decision']['winner'],'prompt')
        self.assertEqual(set(self.data['stages']),{'e2-legacy','e2-prompt'})
        self.assertEqual(set(self.data['lineage']['model_artifacts']),{'exl3-405.json','vision-bf16.json'})
        self.assertTrue(all(a['engine_config']==self.arms[0]['engine_config'] for a in self.arms))
        self.assertEqual(self.arms[0]['environment']['QWEN_PREFILL_KERNEL'],'0')
        self.assertEqual(self.arms[1]['environment']['QWEN_PREFILL_KERNEL'],'1')

    def test_publication_has_no_other_experiment_narrative_or_private_locations(self):
        for name in ('README.md','results/prefill-repeat.md','results/prefill-repeat.json'):
            text=(ROOT/name).read_text()
            self.assertNotRegex(text,r'(?i)ng8|sparkfold|round-2026-10|/home/|/opt/|/run/|\.hermes|bearer\s|api[-_]key')
        self.assertEqual(self.data['policy']['qualification']['reps'],3)

    def test_fixed_denominators_and_eligibility(self):
        for a in self.arms:
            self.assertEqual(a['primary_requests'],33)
            self.assertEqual(a['actual_counts'],a['expected_counts'])
            self.assertEqual(a['actual_counts'],{'speed':12,'decode_long':3,'prefill':18,'quality':0,'witness':0,'total':33})
            self.assertEqual(len({r['request_id'] for r in a['requests']}),33)
            self.assertTrue(a['eligible'])
            self.assertTrue(all(r['eligible'] and not r['errors'] and r['overshoot_tokens']==0 for r in a['requests']))
            self.assertEqual(a['quality']['status'],'not_run')

    def test_all_33_paired_streams_and_hashes_match(self):
        identity=self.data['token_identity']
        self.assertEqual(identity['paired_requests'],33)
        self.assertEqual(identity['identical_requests'],33)
        self.assertEqual(identity['missing_sides'],0)
        self.assertEqual(len(identity['pairs']),33)
        for a,b in zip(self.arms[0]['requests'],self.arms[1]['requests']):
            self.assertEqual(a['request_id'],b['request_id'])
            self.assertEqual(a['raw_output_token_ids'],b['raw_output_token_ids'])
            for r in (a,b):
                self.assertEqual(digest(r['raw_output_token_ids']),r['raw_output_token_ids_sha256'])
                self.assertEqual(len(r['raw_output_token_ids']),r['completion_tokens'])
        self.assertEqual(sum(len(r['raw_output_token_ids']) for r in self.arms[0]['requests']),6018)

    def test_raw_clocks_numerators_and_native_rates(self):
        for a in self.arms:
            for r in a['requests']:
                chunks=r['emission_chunks']
                self.assertEqual(sum(c['n'] for c in chunks),r['raw_completion_tokens'])
                self.assertEqual(chunks[0]['n'],r['first_chunk_tokens'])
                self.assertEqual(r['decode_tokens'],r['raw_completion_tokens']-r['first_chunk_tokens'])
                self.assertAlmostEqual(r['ttft_seconds'],(r['first_emission_ns']-r['request_start_ns'])/1e9)
                self.assertAlmostEqual(r['elapsed_seconds'],(r['request_end_ns']-r['request_start_ns'])/1e9)
                self.assertAlmostEqual(r['prefill_tps_first_sample_inclusive'],r['prompt_tokens']/r['prefill_seconds'])
                self.assertEqual(r['engine_stats']['cached'],0)
                if r['decode_tokens']:
                    self.assertAlmostEqual(r['decode_seconds'],(r['last_emission_ns']-r['first_emission_ns'])/1e9)
                    self.assertAlmostEqual(r['decode_tps'],r['decode_tokens']/r['decode_seconds'])

    def test_every_cell_is_the_median_of_three_real_rows(self):
        fields={'decode_median_tps':'decode_tps','ttft_median_seconds':'ttft_seconds','prefill_median_tps_first_sample_inclusive':'prefill_tps_first_sample_inclusive'}
        for a in self.arms:
            self.assertEqual(len(a['cells']),11)
            for c in a['cells']:
                rows=[r for r in a['requests'] if (r['suite'],r['fixture'])==(c['suite'],c['fixture'])]
                self.assertEqual(len(rows),3)
                self.assertEqual(c['repetitions'],3)
                for summary,raw in fields.items():
                    values=[r[raw] for r in rows]
                    expected=statistics.median(values) if all(v is not None for v in values) else None
                    self.assertEqual(c[summary],expected)

    def test_five_decode_cells_meet_the_predeclared_gate(self):
        comparisons=self.data['decision']['comparisons']
        self.assertEqual(len(comparisons),5)
        for c in comparisons:
            self.assertAlmostEqual(c['change_fraction'],c['after']/c['before']-1)
            self.assertLessEqual(abs(c['change_fraction']),0.03)
        self.assertFalse(self.data['decision']['output_mismatches'])

    def test_readme_numbers_are_derived_from_the_receipt(self):
        text=(ROOT/'README.md').read_text()
        for before in self.arms[0]['cells']:
            after=next(c for c in self.arms[1]['cells'] if (c['suite'],c['fixture'])==(before['suite'],before['fixture']))
            if before['suite']=='prefill':
                gain=100*(after['prefill_median_tps_first_sample_inclusive']/before['prefill_median_tps_first_sample_inclusive']-1)
                self.assertIn(f"+{gain:.1f}%",text)
                for c in (before,after):
                    self.assertIn(f"{c['ttft_median_seconds']:.3f} s",text)
                    self.assertIn(f"{c['prefill_median_tps_first_sample_inclusive']:.2f} tok/s",text)
        self.assertIn('no cold-performance claim',text)
        self.assertIn('not a decode-speed claim',text)

    def test_inventory_and_patch_pins_are_unchanged(self):
        lineage=self.data['lineage']
        self.assertEqual(lineage['ng6_revision'],'55a732e0c4c3d4614bc42b68493bb930d9b02c0a')
        self.assertEqual(lineage['tensorfold_revision'],'609ca419abecebdc5a059498a613680bd3aa847f')
        for name,info in lineage['model_artifacts'].items():
            manifest=json.loads((ROOT/'manifests'/name).read_text())
            self.assertEqual(manifest['revision'],info['revision'])
            self.assertEqual(digest(manifest['files']),info['file_inventory_canonical_sha256'])
        for name,sha in lineage['recipe_patches'].items():
            self.assertEqual(hashlib.sha256((ROOT/'patches'/name).read_bytes()).hexdigest(),sha)

    def test_original_safety_floors_and_observed_extrema(self):
        p=self.data['policy']
        self.assertEqual((p['minimum_MemAvailable'],p['minimum_disk_free'],p['maximum_container_swap'],p['maximum_net_host_swap_growth']),(7*1024**3,20*1024**3,0,1024**3))
        for a in self.arms:
            g=a['guard']
            self.assertTrue(g['completed'])
            self.assertEqual(g['returncode'],0)
            self.assertGreaterEqual(g['minimum_MemAvailable'],p['minimum_MemAvailable'])
            self.assertGreaterEqual(g['minimum_disk_free'],p['minimum_disk_free'])
            self.assertEqual(g['maximum_container_swap'],0)
            self.assertLessEqual(g['maximum_host_swap_growth'],p['maximum_net_host_swap_growth'])

    def test_local_links_resolve(self):
        for name in ('README.md','results/prefill-repeat.md'):
            p=ROOT/name
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
                if '://' in target or target.startswith('#'):continue
                self.assertTrue((p.parent/target.split('#')[0]).is_file(),target)

    def test_fixture_hashes_and_documented_64k_construction(self):
        for a in self.arms:
            for r in a['requests']:
                fixture=a['fixtures'][r['fixture']]
                for key in ('prompt_sha256','prompt_token_ids_sha256'):
                    self.assertEqual(r[key],fixture[key])
                    self.assertRegex(r[key],r'^[0-9a-f]{64}$')
            self.assertFalse(a['fixtures']['prefill/65536-code']['provenance']['retained_64k_found'])
        self.assertIn('extend retained 32K text by repetition',(ROOT/'results/prefill-repeat.md').read_text())

if __name__=='__main__':unittest.main()
