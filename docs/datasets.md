# Dataset and license

The trained time/space classifiers use [TASTY](https://huggingface.co/datasets/Banana-Leopard/TASTY), from [Moudgalya et al. (2023)](https://arxiv.org/abs/2305.05379). The adapter pins revision `8c273d0476910f9b08b8cf01c2e31fd843ee2d7f` and raw SHA256 `ff746b37b2d7adf8e7436c772732c62a2385b18b4e78b5ad1ef518e96f70b61c`.

The dataset card declares CC-BY-3.0; its narrative says CCBY. The authors describe scraped GeeksforGeeks snippets with manual correction. We retain attribution and provenance, do not redistribute downloaded snippets, and do not assert independent verification of every underlying snippet’s rights. Attribution: Kaushik Moudgalya, Ankit Ramakrishnan, Vamsikrishna Chemudupati and Xing Han Lu; Banana-Leopard/TASTY; underlying GeeksforGeeks content.

There are 1,410 C++ rows. Time labels retain 1,382 rows, excluding 28 `np` values because NP does not establish exponential runtime. Space labels retain 1,396 rows, excluding 14 `np` values. Comments are removed before modeling. Coarse labels collapse some multidimensional complexities, so task assumptions matter.

Original problem URLs are unavailable. Descriptive leading-comment signatures and normalized clone grouping approximate problem families; they cannot establish fully independent source generalization. An expert-labeled independent C++ corpus remains needed. CodeNet problem difficulty/runtime and Java/Python-only complexity corpora are not substitutes for C++ asymptotic ground truth.

Raw datasets, source rows and row-level predictions stay ignored; source/provenance hashes are versioned.
