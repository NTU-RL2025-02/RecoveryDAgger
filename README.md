<a id="readme-top"></a>

<div align="center">
  
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT license][license-shield]][license-url]

<h1 align="center">RecoveryDAgger</h1>

  <p align="center">
    Query-Efficient Online Imitation Learning Through Recovery Policy
    <br /> <br />
    <a href="https://github.com/NTU-RL2025-02/RecoveryDAgger/blob/main/Report.pdf">Paper</a>
    &middot;
    <a href="https://docs.google.com/presentation/d/1ntnNjOAhUretADrlI4ycZFx2BClxzAWE0GYV3HPjv28/edit?usp=sharing">Slides</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About RecoveryDAgger

Imitation learning enables agents to acquire complex behaviors from expert demonstrations, yet standard behavior cloning (BC) often suffers from covariate shift
and compounding errors in sequential decision-making tasks. Interactive methods
such as DAgger alleviate this issue by querying the expert on states visited by the
learner, but they typically require frequent expert supervision, resulting in high
annotation cost.

In this work, we propose RecoveryDAgger, a query-efficient interactive imitation
learning framework that augments DAgger-style training with a learned recovery
mechanism. Instead of immediately querying the expert in risky states, RecoveryDAgger first invokes a recovery policy that locally corrects the agent’s behavior
by ascending the gradient of a learned Success Q-function, which estimates the
probability of task completion. Expert queries are reserved for truly novel states
where recovery is unreliable, thereby reducing redundant supervision. Experiments
on the PointMaze navigation task demonstrate that RecoveryDAgger significantly
reduces the number of expert queries while achieving comparable success rates to
strong query-efficient baselines. Our work establishes the effectiveness of integrating learned recovery policies into interactive imitation learning to enhance query
efficiency.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

Before installing the project, make sure you have the following dependencies installed:

- **Python ≥ 3.10**
- One of the following environment managers:

  - **Conda** (Anaconda or Miniconda) _(recommended)_
  - **Python venv**

### Installation

Follow the steps below to set up the environment and install the project.

#### Option A: Using Conda (Recommended)

##### 1. Clone the repository

```sh
git clone https://github.com/NTU-RL2025-02/RecoveryDAgger.git
cd RecoveryDAgger
```

Alternatively, you may download the source code from the **Releases** section.

##### 2. Create and activate a Conda environment

```sh
conda create -n recoverydagger python=3.10
conda activate recoverydagger
```

##### 3. Install PyTorch

Install PyTorch according to your platform.
Please refer to the official PyTorch website for the latest instructions:

👉 [https://pytorch.org](https://pytorch.org)

Examples:

```sh
# CPU / Apple Silicon
pip install torch torchvision
```

```sh
# CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

##### 4. Install project dependencies

From the repository root directory:

```sh
pip install -e recoverydagger
```

---

#### Option B: Using Python venv (Without Conda)

<details>
If Conda is not available, you can use Python’s built-in `venv` module instead.

##### 1. Clone the repository

```sh
git clone https://github.com/NTU-RL2025-02/RecoveryDAgger.git
cd RecoveryDAgger
```

##### 2. Create and activate a virtual environment

```sh
python3 -m venv venv
```

Activate the environment:

- **macOS / Linux**

  ```sh
  source venv/bin/activate
  ```

- **Windows**

  ```sh
  venv\Scripts\activate
  ```

##### 3. Upgrade pip (recommended)

```sh
pip install --upgrade pip
```

##### 4. Install PyTorch

Install PyTorch according to your platform:

👉 [https://pytorch.org](https://pytorch.org)

(Use the same commands as in Option A.)

##### 5. Install project dependencies

```sh
pip install -e .
```

</details>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

> **Tip:** All Python scripts support the `--help` flag.  
> You can run `python <script>.py --help` to see all available options and their descriptions.

This section describes how to collect data, train models, and evaluate trained policies.

### Data Collection

1. A pre-generated offline demonstration dataset is provided at:

   ```
   models/demonstrations/offline_data_100.pkl
   ```

2. If you wish to regenerate the offline dataset from scratch, please refer to the data collection scripts.

<!-- ```sh
python models/demonstrations/gen_offline_data_maze.py --rule-base-expert [--episodes] [--max_steps] [--output ][--deterministic][--seed] [--min_return ]
``` -->

### Training

#### 1. Train RecoveryDAgger from scratch

This command trains RecoveryDAgger starting from behavior cloning (BC) pretraining.

```sh
python3 train.py \
    --seed 48763 \
    --device 0 \
    --iters 100 \
    --demonstration_set_file "models/demonstrations/offline_data_100.pkl" \
    --environment "PointMaze_4rooms-v3" \
    --recovery_type "q" \
    --num_test_episodes 100 \
    --noisy_scale 1.0 \
    --save_bc_checkpoint "models/bc_models/4room_rule_base_100.pt" \
    --fix_thresholds \
    sample_experiment
```

#### 2. Continue training from a pretrained BC model

If you already have a pretrained behavior cloning model, you can skip BC pretraining and continue training directly:

```sh
python3 train.py \
    --seed 48763 \
    --device 0 \
    --iters 100 \
    --demonstration_set_file "models/demonstrations/offline_data_100.pkl" \
    --environment "PointMaze_4rooms-v3" \
    --recovery_type "q" \
    --num_test_episodes 100 \
    --fix_thresholds \
    --noisy_scale 1.0 \
    --skip_bc_pretrain \
    --bc_checkpoint "models/bc_models/4rooms_rule_base_100_noise_0.pt" \
    sample_load_bc_experiment
```

**Notes:**

- `--recovery_type`: Specifies the type of recovery policy to use.
  Supported options:

  - `"q"` (Success Q)
  - `"five_q"` (Ensemble Success Q)
  - `"expert"` (ThriftyDAgger baseline)

- `--demonstration_set_file`: Path to the offline demonstration dataset.

- `--skip_bc_pretrain`: Skip behavior cloning (BC) pretraining and start training directly from a pretrained BC model.

- `--bc_checkpoint`: Path to the pretrained BC model checkpoint.
  This flag is required when `--skip_bc_pretrain` is set.
- After training, the trained model will located at `data/[exp_name]/[exp_name]_s[seed]/best_model.pt`

### Evaluation

To evaluate a trained model:

```sh
python eval.py data/sample/best_model.pt \
    --environment "PointMaze_4rooms-v3" \
    --iters 100
```

This script runs multiple evaluation episodes and reports performance metrics.
Optional visualization flags (e.g., rendering or Q-value heatmaps) can be enabled if supported.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- []()
- []()
- []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/NTU-RL2025-02/RecoveryDAgger.svg?style=for-the-badge
[contributors-url]: https://github.com/NTU-RL2025-02/RecoveryDAgger/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/NTU-RL2025-02/RecoveryDAgger.svg?style=for-the-badge
[forks-url]: https://github.com/NTU-RL2025-02/RecoveryDAgger/network/members
[stars-shield]: https://img.shields.io/github/stars/NTU-RL2025-02/RecoveryDAgger.svg?style=for-the-badge
[stars-url]: https://github.com/NTU-RL2025-02/RecoveryDAgger/stargazers
[issues-shield]: https://img.shields.io/github/issues/NTU-RL2025-02/RecoveryDAgger.svg?style=for-the-badge
[issues-url]: https://github.com/NTU-RL2025-02/RecoveryDAgger/issues
[license-shield]: https://img.shields.io/github/license/NTU-RL2025-02/RecoveryDAgger.svg?style=for-the-badge
[license-url]: https://github.com/NTU-RL2025-02/RecoveryDAgger/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png

<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->

[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
