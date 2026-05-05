# Verified References

All citations below were verified against primary sources via web search before
being committed to the rebuild design. Last verified: 2026-05-05.

When using these in the manuscript, copy the BibTeX into `references.bib`.

## Datasets and benchmarks

### FLamby — Ogier du Terrail et al., 2022
- **Title:** FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings
- **Venue:** NeurIPS 2022 (Datasets & Benchmarks track)
- **arXiv:** 2210.04620
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2022/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets_and_Benchmarks.html
- **Use in our paper:** Source for Fed-ISIC2019 (Setting A) and Fed-Heart-Disease (Setting B). Reports natural site-level heterogeneity.

### LEAF — Caldas et al., 2018
- **Title:** LEAF: A Benchmark for Federated Settings
- **Venue:** Workshop on Federated Learning for Data Privacy and Confidentiality (NeurIPS 2019 workshop)
- **arXiv:** 1812.01097
- **Authors:** Caldas, Duddu, Wu, Li, Konečný, McMahan, Smith, Talwalkar
- **Use:** Cited if FEMNIST is brought back (currently not in design). Documenting in case Setting D is needed.

### Hsu et al., 2019 (Dirichlet partitioning)
- **Title:** Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification
- **arXiv:** 1909.06335 (preprint, no formal venue)
- **Authors:** Hsu, Qi, Brown
- **Use:** Source for Dirichlet$(\alpha)$ partitioning in Setting C.

## Real-world FL deployments

### Sheller et al., 2020
- **Title:** Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data
- **Venue:** Scientific Reports 10:12598
- **DOI:** 10.1038/s41598-020-69250-1
- **Use:** Cited as motivation for Setting A — multi-institutional medical FL is real.

### Dayan et al., 2021
- **Title:** Federated learning for predicting clinical outcomes in patients with COVID-19
- **Venue:** Nature Medicine 27(10):1735–43
- **Use:** Cited as motivation for Setting A — 20 hospitals, real heterogeneity, EXAM model AUC > 0.92.

### Heyndrickx et al., 2024 (MELLODDY)
- **Title:** MELLODDY: Cross-pharma Federated Learning at Unprecedented Scale Unlocks Benefits in QSAR without Compromising Proprietary Information
- **Venue:** Journal of Chemical Information and Modeling (2024). **Not Nature** despite occasional misattribution.
- **Use:** Cited as motivation for industry consortium FL with structural heterogeneity.

## Federated learning architectures

### Bonawitz et al., 2019
- **Title:** Towards Federated Learning at Scale: System Design
- **Venue:** SysML 2019 (now MLSys)
- **arXiv:** 1902.01046
- **Use:** Cited for production FL system design context.

### Liu et al., 2020
- **Title:** Client-Edge-Cloud Hierarchical Federated Learning
- **Venue:** ICC 2020 (IEEE International Conference on Communications), Dublin. **Conference, not journal.**
- **DOI:** 10.1109/ICC40277.2020.9148862
- **Authors:** Liu, Zhang, Song, Letaief
- **Use:** Cited as architectural motivation for hierarchical FL in Setting A.

### Abad et al., 2020
- **Title:** Hierarchical Federated Learning Across Heterogeneous Cellular Networks
- **Venue:** ICASSP 2020, pp. 8866–8870
- **arXiv:** 1909.02362
- **Authors:** Abad, Ozfatura, Gunduz, Ercetin
- **Use:** Cited as supporting evidence for hierarchical FL deployment.

### Roy et al., 2019 — BrainTorrent
- **Title:** BrainTorrent: A Peer-to-Peer Environment for Decentralized Federated Learning
- **Venue:** MICCAI 2019
- **arXiv:** 1905.06731
- **Authors:** Roy, Siddiqui, Pölsterl, Navab, Wachinger
- **Use:** Cited as motivation for decentralized P2P FL in Setting B.

### Hegedűs et al., 2021
- **Title:** Decentralized learning works: An empirical comparison of gossip learning and federated learning
- **Venue:** Journal of Parallel and Distributed Computing 148:109–124
- **Authors:** Hegedűs, Danner, Jelasity
- **Use:** Cited for decentralized/gossip FL comparison.

### Briggs et al., 2020
- **Title:** Federated learning with hierarchical clustering of local updates to improve training on non-IID data
- **Venue:** IJCNN 2020 (Glasgow, July 19–24, part of WCCI 2020)
- **arXiv:** 2004.11791
- **Authors:** Briggs, Fan, András
- **Use:** Cited in Related Work as a hierarchical clustering scheme **and** as a critique target — Briggs et al.'s hierarchy is constructed from data, so the leakage we study is by design in their setting.

### Beilharz et al., 2021
- **Title:** Implicit Model Specialization through DAG-based Decentralized Federated Learning
- **Venue:** ACM Middleware 2021
- **arXiv:** 2111.01257
- **Use:** Cited in Related Work as DAG-decentralized FL example.

### Wang et al., 2021 — Field Guide
- **Title:** A Field Guide to Federated Optimization
- **arXiv:** 2107.06917 (53 authors, lead Wang)
- **Use:** Cited as authoritative survey on FL formulation, relevant for Sections 4.4 (cross-silo) and 4.5 (decentralized).

## Privacy attacks

### Shokri et al., 2017
- **Title:** Membership Inference Attacks Against Machine Learning Models
- **Venue:** IEEE S&P 2017, pp. 3–18
- **arXiv:** 1610.05820
- **Authors:** Shokri, Stronati, Song, Shmatikov
- **Use:** Source of the **shadow-model framework** that TADI relies on for offline training.

### Melis et al., 2019
- **Title:** Exploiting Unintended Feature Leakage in Collaborative Learning
- **Venue:** IEEE S&P 2019, pp. 691–706
- **arXiv:** 1805.04049
- **Authors:** Melis, Song, De Cristofaro, Shmatikov
- **Use:** Established prior work on parameter-sequence distributional leakage. TADI's parameter-only ablation $\mathcal{A}_1$ is the spiritual successor.

### Zhu et al., 2019 — DLG
- **Title:** Deep Leakage from Gradients
- **Venue:** NeurIPS 2019
- **arXiv:** 1906.08935
- **Authors:** Zhu, Liu, Han
- **Use:** Cited as the original gradient-inversion attack. Used as comparator baseline (run at our DP levels).

### Geiping et al., 2020
- **Title:** Inverting Gradients — How easy is it to break privacy in federated learning?
- **Venue:** NeurIPS 2020
- **arXiv:** 2003.14053
- **Authors:** Geiping, Bauermeister, Dröge, Moeller
- **Use:** Stronger gradient-inversion comparator. **Important:** does **not** characterize DP-SGD breakdown — we measure that empirically rather than citing them for it.

## Theoretical machinery (Stage 4)

### Mironov, 2017 — Rényi Differential Privacy
- **Title:** Rényi Differential Privacy
- **Venue:** IEEE Computer Security Foundations Symposium (CSF) 2017, pp. 263–275
- **arXiv:** 1702.07476
- **Use:** Foundation for the Stage 4 MI bound. Provides per-mechanism RDP for the Gaussian mechanism, composition rules, and conversion to $(\varepsilon, \delta)$-DP.

### Wang, Balle, Kasiviswanathan, 2019 — Subsampled RDP
- **Title:** Subsampled Rényi Differential Privacy and Analytical Moments Accountant
- **Venue:** AISTATS 2019, pp. 1226–1235 (Notable Paper Award)
- **arXiv:** 1808.00087
- **Use:** Tight RDP bound for client subsampling. Used if Stage 4 incorporates the subsampling layer; not load-bearing for the base theorem.

### Cuff & Yu, 2016 — DP as MI constraint
- **Title:** Differential Privacy as a Mutual Information Constraint
- **Venue:** ACM CCS 2016, pp. 43–54 (Vienna)
- **arXiv:** 1608.03677
- **Authors:** Cuff, Yu (Princeton)
- **Use:** Establishes MI-DP as an intermediate notion sandwiched between $\varepsilon$-DP and $(\varepsilon, \delta)$-DP. Foundational for the RDP-to-MI conversion underlying Theorem 1.

### Asoodeh et al., 2021 — Three Variants of DP
- **Title:** Three Variants of Differential Privacy: Lossless Conversion and Applications
- **Venue:** **IEEE J. Sel. Areas Inf. Theory** vol. 2(1), DOI 10.1109/JSAIT.2021.3054692
- **arXiv:** 2008.06529
- **Use:** Sharper RDP-to-DP conversion; alternative (and tighter) route to the MI bound. Backup if the Cuff-Yu route gives loose constants.

### Abadi et al., 2016 — Deep Learning with DP / DP-SGD
- **Title:** Deep Learning with Differential Privacy
- **Venue:** ACM CCS 2016
- **arXiv:** 1607.00133
- **Use:** Foundation of DP-SGD, moments accountant. Cited in Proposition 1 as the DP-SGD foundation.

### McMahan et al., 2018 — DP-FedAvg
- **Title:** Learning Differentially Private Recurrent Language Models
- **Venue:** ICLR 2018
- **arXiv:** 1710.06963
- **Authors:** McMahan, Ramage, Talwar, Zhang
- **Use:** Per-client DP-FedAvg analysis matching our setup. Primary citation for Proposition 1's per-client noise formulation.

### Wei et al., 2020 — DP-FL convergence (NbAFL)
- **Title:** Federated Learning with Differential Privacy: Algorithms and Performance Analysis
- **Venue:** IEEE Trans. Information Forensics and Security 2020, pp. 3454–3469
- **DOI:** 10.1109/TIFS.2020.2988575
- **Use:** Convergence bound form for DP-FL. **Caveat:** their NbAFL noises the aggregate, not per-client. We adapt the bound form to per-client.

### Li et al., 2020 — FedAvg convergence on non-IID
- **Title:** On the Convergence of FedAvg on Non-IID Data
- **Venue:** ICLR 2020
- **arXiv:** 1907.02189
- **Use:** Standard $O(1/T)$ convergence rate for strongly convex non-IID FedAvg. Cited in Proposition 1 for non-IID convergence machinery.

### Bonawitz et al., 2017 — Secure aggregation
- **Title:** Practical Secure Aggregation for **Privacy-Preserving Machine Learning** (CCS title — the *"…User-Held Data"* title is the earlier arXiv preprint)
- **Venue:** ACM CCS 2017, pp. 1175–1191
- **arXiv:** 1611.04482
- **Use:** Cited in §3.6 secure-aggregation extension experiment. Not load-bearing for the main theorem.
