<div align="center">
  DQNC2S: DQN-based Cross-stream Crisis event Summarizer
</div>

<div align="center">
<br />

[![Project license](https://img.shields.io/github/license/DarthReca/crisis-dqn.svg?style=flat-square)](LICENSE)

[![Pull Requests welcome](https://img.shields.io/badge/PRs-welcome-ff69b4.svg?style=flat-square)](https://github.com/DarthReca/crisis-dqn/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
[![code with love by DarthReca](https://img.shields.io/badge/%3C%2F%3E%20with%20%E2%99%A5%20by-DarthReca-ff1414.svg?style=flat-square)](https://github.com/DarthReca)

</div>

<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
- [Dataset](#dataset)
- [Usage](#usage)
- [Authors & contributors](#authors--contributors)
- [License](#license)
- [Acknowledgements](#acknowledgements)

</details>

---

## About

This is the source code for experiments of **DQNC2S: DQN-based Cross-stream Crisis event Summarizer** paper accepted at **ECIR 2024**.

**The repository is in the creation phase; some files could be missing**

## Dataset

The employed dataset is *CrisisFACTS 2022*.

To download the dataset, refer to https://github.com/crisisfacts/utilities/tree/main and the [competition website](https://crisisfacts.github.io/). 

## Usage

The module *environment* contains a Gymnasium Env. It could be used with different frameworks. 

The suggested environment is `SimilarityCrisisEnv`, which includes the BERT embedding, max similarity, and remaining space. The following are its attributes:

**Action Space**: `Discrete(2)`

**Observation Space**:  `Box(770, N)`



The `requests_file` ipc should have the following schema: 

- **('eventID', String)**

- **('requestID', String)**

- **('dateString', String)**

Each file with annotations contained in `data_folder` should have the following schema:

- **('text', String),** the passage

- **('source_type', String)**

- **('unix_timestamp', Int64)**

- **('answer', String)**, the answer to the query given text

- **('score', Float64),**, confidence for the answer

- **('query', String)**, the query 

To better understand the meaning, refer to **CrisisFacts 2022** data. 

## Authors & contributors

The original setup of this repository is by [Daniele Rege Cambrin](https://github.com/DarthReca).

For a full list of all authors and contributors, see [the contributors page](https://github.com/DarthReca/crisis-dqn/contributors).

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

The project makes use of CleanRL under the **MIT license**. See [CLEANRL_LICENSE](CLEANRL_LICENSE) for more information.

# 
