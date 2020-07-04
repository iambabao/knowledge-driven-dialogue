# knowledge-driven-dialogue

Code for the project [knowledge-driven-dialogue](https://github.com/baidu/knowledge-driven-dialogue)

This repo re-implements the [Pointer-generator network](https://www.aclweb.org/anthology/P17-1099.pdf) for the task with tensorflow==1.14.

## models

- seq2seq: traditional Seq2Seq model with attention mechanism.
- ptrnet_h: Seq2Seq model with pointer mechanism over the conversation history.
- ptrnet_k: Seq2Seq model with pointer mechanism over the knowledge.
- dual_ptrnet: Seq2Seq model with pointer mechanism over both the conversation history and the knowledge.
