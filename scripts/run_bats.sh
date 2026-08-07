# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python src/agent_bats.py --model gemini/gemini-2.5-pro \
        --search_budget 100 --browse_budget 100 \
        --begin 0 --end -1 \
        --max_output_tokens 65535 --temperature 0.7 \
        --concurrency 15 \
        --tag bc200_bats \
        --do_eval \
        --flag_optimize_query --flag_browse_flash --flag_gemini_no_think --flag_remove_tool_responses --flag_cont_gen \
        --none_ans_threshold 3 \
        --n_query_results 10 \
        --sum_frequency 10 \
        --data_path data/browsecomp/bc200.jsonl \
        --seed 10