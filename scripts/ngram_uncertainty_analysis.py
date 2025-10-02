"""Self-contained pipeline for dataset construction and n-gram uncertainty analysis.

This script creates a synthetic biography dataset, simulates power-law sampling
regimes, computes the Monofact rate, estimates subjective uncertainty via an
n-gram language model, and writes both a CSV summary and an SVG scatter plot that
relates the two metrics.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Sequence, Tuple

PROMPT_TOKEN = "<BIOGRAPHY>"
DEFAULT_OUTPUT_DIR = Path("outputs")
SVG_WIDTH = 900
SVG_HEIGHT = 600
SVG_MARGIN = 70

# Deterministic pools of attributes (avoids external downloads)
NAME_POOL = [
    "Noah Rhodes", "Angie Henderson", "Daniel Wagner", "Cristian Santos", "Connie Lawrence",
    "Avery Morgan", "Jordan Ellis", "Taylor Brooks", "Riley Bennett", "Emerson Clarke",
    "Sydney Barrett", "Morgan Hayes", "Parker Quinn", "Rowan Blake", "Elliot Mason",
    "Avery Patel", "Jordan Chen", "Taylor Gupta", "Riley Singh", "Emerson Alvarez",
    "Quinn Ramsey", "Dakota Miles", "Skyler Cruz", "Harper Lane", "Reese Nolan",
    "Sawyer Tate", "Logan Avery", "Cameron Reid", "Payton Sloan", "Spencer Vega",
]
CITY_POOL = [
    "Greenwood", "Hueytown", "Oriska", "Fowler", "Dysart", "Riverside", "Kingston",
    "Fairview", "Oakridge", "Mapleton", "Summit", "Lakeside", "Hillcrest", "Cedar Grove",
    "Brookfield", "Silverton", "Westhaven", "Northbridge", "Highland", "Grandview",
]
COMPANY_POOL = [
    "Brightwave Analytics", "Silverline Studios", "Northwind Ventures", "Blue Horizon Labs",
    "Evergreen Media", "Cascade Robotics", "Solaris Systems", "Apex Consulting",
    "Lighthouse Partners", "Summit Works", "Aurora Health", "Vertex Dynamics",
]
JOB_POOL = [
    "data scientist", "software engineer", "product manager", "research analyst",
    "marketing director", "operations lead", "financial advisor", "UX designer",
    "biomedical engineer", "policy strategist", "creative producer", "systems architect",
]
DEGREE_POOL = [
    "Computer Science", "Economics", "Mechanical Engineering", "Art History",
    "Biology", "Political Science", "Sociology", "Physics", "Mathematics", "Philosophy",
]
UNIVERSITY_POOL = [
    "Harvard University", "Stanford University", "MIT", "UC Berkeley",
    "Princeton University", "Yale University", "Columbia University", "University of Oxford",
]


@dataclass
class SyntheticBiography:
    prompt: str
    answer: str
    name: str
    gold: List[str]

    def as_record(self) -> Dict[str, object]:
        return {"x": self.prompt, "y": self.answer, "names": self.name, "gold": tuple(self.gold)}


def random_birthdate(rng: random.Random) -> str:
    year = rng.randint(1940, 2002)
    month = rng.randint(1, 12)
    if month == 2:
        day = rng.randint(1, 28)
    elif month in {4, 6, 9, 11}:
        day = rng.randint(1, 30)
    else:
        day = rng.randint(1, 31)
    return f"{month:02d}/{day:02d}/{year}"


def generate_biography(name: str, rng: random.Random) -> SyntheticBiography:
    birthdate = random_birthdate(rng)
    city = rng.choice(CITY_POOL)
    company = rng.choice(COMPANY_POOL)
    job = rng.choice(JOB_POOL)
    degree = rng.choice(DEGREE_POOL)
    university = rng.choice(UNIVERSITY_POOL)

    facts = [
        f"{name} was born on {birthdate}.",
        f"{name} grew up in {city}.",
        f"{name} studied {degree} at {university}.",
        f"{name} works as a {job} at {company}.",
    ]
    answer = " ".join(facts)
    gold = [name, birthdate, city, degree, job, company]
    return SyntheticBiography(prompt=PROMPT_TOKEN, answer=answer, name=name, gold=gold)


def build_base_dataset(size: int, seed: int = 1217) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    names = NAME_POOL.copy()
    rng.shuffle(names)

    records: List[Dict[str, object]] = []
    while len(records) < size:
        name = names[len(records) % len(names)]
        biography = generate_biography(name, rng)
        records.append(biography.as_record())
    return records


def serialise_gold(gold: Sequence[str]) -> str:
    return " | ".join(gold)


def save_dataset(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["x", "y", "names", "gold"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "x": record["x"],
                    "y": record["y"],
                    "names": record["names"],
                    "gold": serialise_gold(record["gold"]),
                }
            )


def create_powerlaw_p(records: List[Dict[str, object]], pareto_alpha: float, rng: random.Random) -> List[Dict[str, object]]:
    expanded: List[Dict[str, object]] = []
    for record in records:
        # Python's paretovariate uses alpha where mean = alpha / (alpha - 1) for alpha > 1
        samples = max(1, int(rng.paretovariate(pareto_alpha)))
        expanded.extend([record] * samples)
    return expanded


def sample_records(records: List[Dict[str, object]], size: int, rng: random.Random) -> List[Dict[str, object]]:
    return [rng.choice(records) for _ in range(size)]


def mono_calc(records: List[Dict[str, object]]) -> float:
    pairs = [(r["x"], r["y"], r["names"], r["gold"]) for r in records]
    counts = Counter(pairs)
    mono = sum(1 for c in counts.values() if c == 1)
    return mono / len(records) if records else 0.0


class NGramLanguageModel:
    def __init__(self, n: int = 3, smoothing: float = 1.0):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.alpha = smoothing
        self.context_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def _tokenise(self, text: str) -> List[str]:
        tokens = text.split()
        return ["<s>"] * (self.n - 1) + tokens + ["</s>"]

    def fit(self, texts: Sequence[str]) -> None:
        for text in texts:
            tokens = self._tokenise(text)
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                target = tokens[i + self.n - 1]
                self.context_counts[context][target] += 1

    def context_distribution(self, context: Tuple[str, ...]) -> Dict[str, float]:
        V = len(self.vocab)
        if V == 0:
            return {}
        counts = self.context_counts.get(context)
        if not counts:
            prob = 1.0 / V
            return {token: prob for token in self.vocab}
        total = sum(counts.values()) + self.alpha * V
        return {token: (counts.get(token, 0) + self.alpha) / total for token in self.vocab}

    def context_entropy(self, context: Tuple[str, ...]) -> float:
        dist = self.context_distribution(context)
        if not dist:
            return 0.0
        return -sum(p * math.log(p) for p in dist.values() if p > 0)

    def sentence_entropy(self, text: str) -> float:
        tokens = self._tokenise(text)
        entropies: List[float] = []
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i : i + self.n - 1])
            entropies.append(self.context_entropy(context))
        return fmean(entropies) if entropies else 0.0


def compute_subjective_uncertainty(model: NGramLanguageModel, texts: Sequence[str]) -> float:
    entropies = [model.sentence_entropy(text) for text in texts]
    return fmean(entropies) if entropies else 0.0


def interpolate_colour(value: float, min_val: float, max_val: float) -> str:
    if math.isclose(max_val, min_val):
        t = 0.5
    else:
        t = (value - min_val) / (max_val - min_val)
        t = min(max(t, 0.0), 1.0)
    r = int(74 + t * (68 - 74))
    g = int(0 + t * (231 - 0))
    b = int(135 + t * (84 - 135))
    return f"#{r:02x}{g:02x}{b:02x}"


def save_scatter_plot(path: Path, results: List[Dict[str, float]]) -> None:
    xs = [row["subjective_uncertainty"] for row in results]
    ys = [row["monofact_rate"] for row in results]
    cs = [row["pareto_alpha"] for row in results]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_c, max_c = min(cs), max(cs)

    width, height = SVG_WIDTH, SVG_HEIGHT
    margin = SVG_MARGIN
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    def scale_x(x: float) -> float:
        if math.isclose(max_x, min_x):
            return margin + plot_width / 2
        return margin + (x - min_x) / (max_x - min_x) * plot_width

    def scale_y(y: float) -> float:
        if math.isclose(max_y, min_y):
            return height - margin - plot_height / 2
        return height - margin - (y - min_y) / (max_y - min_y) * plot_height

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text { font-family: Arial, sans-serif; }</style>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' stroke-width='2' />",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' stroke-width='2' />",
        f"<text x='{width / 2}' y='{margin / 2}' text-anchor='middle' font-size='20'>Subjective uncertainty vs. Monofact rate</text>",
        f"<text x='{width / 2}' y='{height - margin / 3}' text-anchor='middle' font-size='18'>Subjective uncertainty (entropy)</text>",
        f"<text x='{margin / 3}' y='{height / 2}' text-anchor='middle' font-size='18' transform='rotate(-90 {margin / 3},{height / 2})'>Monofact rate</text>",
    ]

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x_val = min_x + frac * (max_x - min_x)
        x_pos = scale_x(x_val)
        y_base = height - margin
        lines.append(f"<line x1='{x_pos}' y1='{y_base}' x2='{x_pos}' y2='{y_base + 8}' stroke='black' stroke-width='1' />")
        lines.append(f"<text x='{x_pos}' y='{y_base + 24}' text-anchor='middle' font-size='14'>{x_val:.2f}</text>")

        y_val = min_y + frac * (max_y - min_y)
        y_pos = scale_y(y_val)
        lines.append(f"<line x1='{margin}' y1='{y_pos}' x2='{margin - 8}' y2='{y_pos}' stroke='black' stroke-width='1' />")
        lines.append(f"<text x='{margin - 12}' y='{y_pos + 5}' text-anchor='end' font-size='14'>{y_val:.2f}</text>")

    for x, y, c in zip(xs, ys, cs):
        colour = interpolate_colour(c, min_c, max_c)
        lines.append(
            f"<circle cx='{scale_x(x)}' cy='{scale_y(y)}' r='6' fill='{colour}' stroke='black' stroke-width='0.5'>"
            f"<title>alpha={c:.2f}\nuncertainty={x:.3f}\nmonofact={y:.3f}</title>"
            "</circle>"
        )

    legend_x = width - margin + 20
    legend_y_start = margin
    legend_height = plot_height
    legend_width = 20
    lines.append(
        f"<text x='{legend_x + legend_width / 2}' y='{legend_y_start - 10}' text-anchor='middle' font-size='16'>Pareto alpha</text>"
    )
    steps = 40
    for i in range(steps):
        frac = i / (steps - 1)
        alpha_val = min_c + frac * (max_c - min_c)
        colour = interpolate_colour(alpha_val, min_c, max_c)
        y0 = legend_y_start + frac * legend_height
        lines.append(
            f"<rect x='{legend_x}' y='{y0}' width='{legend_width}' height='{legend_height / steps + 1}' fill='{colour}' stroke='none' />"
        )
    lines.append(f"<text x='{legend_x + legend_width + 10}' y='{legend_y_start + 5}' font-size='14'>{max_c:.2f}</text>")
    lines.append(f"<text x='{legend_x + legend_width + 10}' y='{legend_y_start + legend_height}' font-size='14'>{min_c:.2f}</text>")
    lines.append("</svg>")

    path.write_text("\n".join(lines), encoding="utf-8")


def pareto_resample(
    base_records: List[Dict[str, object]],
    alphas: Iterable[float],
    sample_size: int,
    seed: int,
) -> Dict[float, List[Dict[str, object]]]:
    datasets: Dict[float, List[Dict[str, object]]] = {}
    for offset, alpha in enumerate(alphas):
        rng = random.Random(seed + offset * 9973)
        expanded = create_powerlaw_p(base_records, pareto_alpha=alpha, rng=rng)
        datasets[alpha] = sample_records(expanded, size=sample_size, rng=rng)
    return datasets


def run_pipeline(
    dataset_size: int,
    pareto_alphas: Sequence[float],
    sample_size: int,
    ngram_order: int,
    smoothing: float,
    output_dir: Path,
    seed: int,
) -> List[Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_records = build_base_dataset(dataset_size, seed)
    dataset_dir = output_dir / "datasets"
    save_dataset(dataset_dir / "base_dataset.csv", base_records)

    ngram = NGramLanguageModel(n=ngram_order, smoothing=smoothing)
    ngram.fit([r["y"] for r in base_records])

    resampled = pareto_resample(base_records, pareto_alphas, sample_size, seed)

    rows: List[Dict[str, float]] = []
    for alpha, records in sorted(resampled.items(), key=lambda item: item[0]):
        save_dataset(dataset_dir / f"pareto_alpha_{alpha:.2f}.csv", records)
        monofact_rate = mono_calc(records)
        subjective_uncertainty = compute_subjective_uncertainty(ngram, [r["y"] for r in records])
        rows.append({
            "pareto_alpha": alpha,
            "monofact_rate": monofact_rate,
            "subjective_uncertainty": subjective_uncertainty,
        })

    csv_path = output_dir / "subjective_uncertainty_vs_monofact.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pareto_alpha", "monofact_rate", "subjective_uncertainty"])
        writer.writeheader()
        writer.writerows(rows)

    save_scatter_plot(output_dir / "subjective_vs_monofact.svg", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the n-gram uncertainty analysis pipeline.")
    parser.add_argument("--dataset-size", type=int, default=200, help="Number of unique biographies to generate.")
    parser.add_argument(
        "--pareto-alphas",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.5, 2.0],
        help="Pareto alpha values used to resample the dataset.",
    )
    parser.add_argument("--sample-size", type=int, default=200, help="Sample size per resampled dataset.")
    parser.add_argument("--ngram-order", type=int, default=3, help="Order of the n-gram language model.")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Laplace smoothing constant.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where outputs are written.")
    parser.add_argument("--seed", type=int, default=1217, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(
        dataset_size=args.dataset_size,
        pareto_alphas=args.pareto_alphas,
        sample_size=args.sample_size,
        ngram_order=args.ngram_order,
        smoothing=args.smoothing,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    for row in results:
        print(
            f"alpha={row['pareto_alpha']:.2f} | monofact={row['monofact_rate']:.3f} | "
            f"uncertainty={row['subjective_uncertainty']:.3f}"
        )
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
