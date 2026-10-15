# Dataset decision record

Research checked 2026-10-16. No third-party source files are committed.

| Primary source | Useful evidence | Decision |
|---|---|---|
| [IBM Project CodeNet](https://github.com/IBM/Project_CodeNet) / [paper](https://arxiv.org/abs/2105.12655) | Large multilingual corpus including C++; metadata `complexity` means problem difficulty, not Big-O. Dataset is CDLA-Permissive-2.0; repository code has its own license. | Good candidate for a future manually annotated C++ subset; cannot honestly treat metadata as complexity ground truth. |
| [CoRCoD](https://github.com/midas-research/corcod-dataset) / [paper](https://arxiv.org/abs/1911.01155) | Runtime-complexity research corpus; Java-oriented, repository exposes archive and metadata. No clear license surfaced in root. | Do not redistribute; clarify license and language adaptation first. |
| [CodeComplex-Data](https://github.com/sybaik1/CodeComplex-Data) / [paper](https://arxiv.org/abs/2401.08719) | Expert labels for Java and Python, not C++. | Useful methodology; cross-language training does not establish C++ generalization. Dataset licensing must be checked independently of model code. |

The included generator produces 28 original C++ teaching examples (four algorithm families per class). These are **synthetic**, openly reproducible from source, and intended only for a smoke benchmark. Their uniform labels and short snippets are unlike online-judge solutions. Do not advertise their metrics as public benchmark performance.

A production corpus should include accepted C++ solutions, explicit input constraints, expert time-complexity annotations, problem IDs, submission provenance, and a clear data license. Review annotation disagreements, detect clone clusters across problems, and freeze a problem-held-out test set before model selection. Training may run on Kaggle once corpus rights, model configuration, and compute requirements are established; this repository needs no account credentials for local smoke runs.

## Real C++ extension: TASTY

[TASTY dataset card](https://huggingface.co/datasets/Banana-Leopard/TASTY) and [author paper](https://arxiv.org/abs/2305.05379) provide C++ space/time labels. The adapter pins revision `8c273d0476910f9b08b8cf01c2e31fd843ee2d7f` and SHA256 `ff746b37b2d7adf8e7436c772732c62a2385b18b4e78b5ad1ef518e96f70b61c`. Card metadata specifies CC-BY-3.0; narrative text only says CCBY. Authors describe scraping GeeksforGeeks with manual correction and collapse multidimensional complexities (e.g. n+m) into coarse classes. We rely on the authors' declared dataset license, retain attribution and provenance, and do not redistribute downloaded snippets or assert independent verification of every underlying snippet's rights.

Attribution: Kaushik Moudgalya, Ankit Ramakrishnan, Vamsikrishna Chemudupati, Xing Han Lu, TASTY (2023), arXiv:2305.05379; Banana-Leopard/TASTY; underlying GeeksforGeeks content.

There are 1,410 C++ rows. The time task retains 1,382 rows, excluding 28 `np` labels because NP does not establish an exponential runtime. The space task retains 1,396 rows, excluding 14 `np` labels. All comments are removed before feature extraction, including comments revealing complexity. The corpus contains no original source URLs or judge problem IDs: we conservatively group repeated descriptive leading-comment signatures and normalized near-duplicate code transitively. Unknown semantic clones and related article variants can remain across partitions; this is an honest grouped corpus evaluation, not a proven unseen-problem benchmark.

No live website scraping, account credentials or Kaggle compute are necessary for this bounded CPU experiment.
