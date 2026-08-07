# BATS: Budget-Aware Tool-Use Enables Effective Agent Scaling

Official codebase for the paper "Budget-Aware Tool-Use Enables Effective Agent Scaling" (accepted at COLM 2026).

[Paper (arXiv)](https://arxiv.org/abs/2511.17006)

## 🏴󠁶󠁵󠁭󠁡󠁰󠁿 Overview

Scaling test-time computation improves performance across tasks for large language models (LLMs) and has been extended to tool-augmented agents, where scaling involves not only “thinking” in tokens but also “acting” via tool calls that directly constrain environmental interaction.
However, simply increasing the tool-call budget fails to improve performance, as agents lack "budget awareness" and quickly hit a performance ceiling.

To address this, we study how to scale such agents effectively under explicit tool-call budgets, focusing on web search agents.
We first introduce the **Budget Tracker**, a lightweight plug-in that provides the agent with continuous budget awareness, enabling simple yet effective scaling.
We further develop **BATS (Budget Aware Test-time Scaling)**, an advanced framework that leverages this awareness to dynamically adapt its planning and verification strategy.

To analyze cost-performance scaling in a controlled manner, we formalize a unified cost metric that jointly accounts for token and tool consumption. We provide the first systematic study on budget-constrained agents, showing that budget-aware methods produce more favorable scaling curves and push the cost-performance Pareto frontier.

## 💬 Citation

If our work is useful for you, please consider citing our paper:

```bibtex
@inproceedings{liu2026budget,
  title={Budget-Aware Tool-Use Enables Effective Agent Scaling},
  author={Liu, Tengxiao and Wang, Zifeng and Miao, Jin and Hsu, I and Yan, Jun and Chen, Jiefeng and Han, Rujun and Xu, Fangyuan and Chen, Yanfei and Jiang, Ke and Daruki, Samira and Liang, Yi and Wang, William Yang and Pfister, Tomas and Lee, Chen-Yu},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```

*This is not an officially supported Google product.*
