# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import sys
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.github/scripts'))
import review_automation


class ReviewAutomationTests(unittest.TestCase):
    def test_generated_csv_is_compacted_but_other_missing_patches_are_rejected(self):
        roster, skill = review_automation.load_config()
        module = next(iter(roster['modules']))
        pr = {'number': 185, 'head': {'sha': '1' * 40},
              'base': {'sha': '2' * 40, 'ref': 'main', 'repo': {'full_name': 't-head/pccl'}},
              'user': {'id': 1}, 'state': 'open', 'draft': False,
              'title': 'Daily CI', 'body': 'Refs #119'}
        generated = {'filename': 'tests/xtest/daily-full-csv/standard.csv', 'status': 'added',
                     'additions': 8900, 'deletions': 0, 'changes': 8900}
        prompt = []
        with mock.patch.object(review_automation, 'ask_copilot',
                               side_effect=lambda value: prompt.append(value) or json.dumps(
                                   {'modules': [module], 'reason': 'CI changes'})):
            review_automation.classify(pr, [generated], roster, skill)
        self.assertIn('<generated CSV payload omitted>', prompt[0])
        with self.assertRaisesRegex(ValueError, 'incomplete or binary diff'):
            review_automation.classification_file({
                'filename': 'src/main.cc', 'status': 'added',
                'additions': 100, 'deletions': 0, 'changes': 100})


if __name__ == '__main__':
    unittest.main()
