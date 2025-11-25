"""
TurnBackHoax Dataset Download Helper

The TurnBackHoax dataset from Mafindo is one of the most comprehensive
Indonesian fake news datasets available.

Sources:
1. Mafindo Official: https://turnbackhoax.id/
2. GitHub Mirror: https://github.com/mafindo/turnbackhoax-dataset (if available)
3. Kaggle: Search for "Indonesian Fake News" or "TurnBackHoax"

This script provides utilities to:
1. Check dataset availability
2. Download from available sources
3. Convert to standard format for training

Manual Download Instructions:
-----------------------------
If automatic download fails, follow these steps:

1. Visit https://turnbackhoax.id/
2. Look for "Data" or "Dataset" section
3. Download the dataset (usually CSV or JSON format)
4. Place in data/turnbackhoax.csv or data/turnbackhoax.json

Expected Format:
----------------
The dataset should have at minimum:
- A text/content column (article content)
- A label column (hoax/valid or similar)

Common column names:
- Text: 'content', 'text', 'narasi', 'article', 'body'
- Label: 'label', 'class', 'kategori', 'status'
"""

import os
import json
import urllib.request
from typing import Optional, Dict, List

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def create_synthetic_turnbackhoax(output_path: str, num_samples: int = 1000) -> str:
    """
    Create a synthetic dataset for initial testing.
    
    This generates Indonesian fake news patterns based on common characteristics
    of hoax news in Indonesia. Use this for testing the pipeline before
    obtaining the real TurnBackHoax dataset.
    
    Args:
        output_path: Path to save the CSV file.
        num_samples: Number of samples to generate.
        
    Returns:
        Path to generated file.
    """
    import random
    import csv
    
    # Hoax patterns (Indonesian fake news characteristics)
    hoax_patterns = [
        ("VIRAL! {topic} ternyata {claim}. Bagikan sebelum dihapus!", 1),
        ("BREAKING: Pemerintah {action} mulai besok. Warga harus waspada!", 1),
        ("Rahasia {entity} terungkap! {claim}.", 1),
        ("AWAS! {item} berbahaya bagi kesehatan menurut dokter!", 1),
        ("Terbukti! {claim}. Video ini membuktikan segalanya.", 1),
        ("{entity} akhirnya mengakui {claim}. Media mainstream menyembunyikan!", 1),
        ("CEK FAKTA: {claim} adalah BENAR kata sumber terpercaya!", 1),
        ("Warga {location} GEGER! {event} terjadi semalam!", 1),
        ("DIBONGKAR! Konspirasi {topic} selama ini disembunyikan!", 1),
        ("Sebarkan! {entity} tidak ingin Anda tahu tentang {topic}!", 1),
    ]
    
    valid_patterns = [
        ("Menteri {ministry} mengumumkan kebijakan baru terkait {topic}.", 0),
        ("Hasil penelitian {institution} menunjukkan {finding}.", 0),
        ("Pemerintah meluncurkan program {program} untuk masyarakat.", 0),
        ("Berdasarkan data BPS, {statistic} pada periode ini.", 0),
        ("Konferensi pers {entity} membahas perkembangan {topic}.", 0),
        ("Laporan tahunan {institution} mencatat {finding}.", 0),
        ("Presiden meresmikan {project} di {location}.", 0),
        ("DPR menyetujui {policy} setelah pembahasan panjang.", 0),
        ("Bank Indonesia melaporkan {statistic} untuk kuartal ini.", 0),
        ("{institution} menggelar {event} di {location}.", 0),
    ]
    
    # Fill-in components
    topics = [
        "vaksin COVID-19", "ekonomi digital", "pendidikan", "kesehatan",
        "teknologi AI", "perubahan iklim", "infrastruktur", "politik"
    ]
    claims = [
        "palsu dan berbahaya", "mengandung zat berbahaya", 
        "menguntungkan elit", "disembunyikan pemerintah",
        "menyebabkan efek samping serius", "sebenarnya adalah penipuan"
    ]
    entities = [
        "WHO", "pemerintah", "perusahaan farmasi", "media besar",
        "Bill Gates", "George Soros", "Illuminati", "elite global"
    ]
    locations = [
        "Jakarta", "Surabaya", "Bandung", "Medan", "Semarang",
        "Makassar", "Palembang", "Denpasar"
    ]
    institutions = [
        "Universitas Indonesia", "ITB", "UGM", "LIPI", "BRIN",
        "Kemenkes", "Bank Indonesia", "BPS"
    ]
    items = [
        "air kemasan", "mie instan", "vaksin", "obat herbal",
        "suplemen", "makanan cepat saji"
    ]
    ministries = ["Kesehatan", "Keuangan", "Pendidikan", "PUPR", "Kominfo"]
    programs = ["bantuan sosial", "digitalisasi", "pelatihan kerja", "beasiswa"]
    findings = ["peningkatan 5%", "penurunan angka kemiskinan", "stabilitas ekonomi"]
    statistics = ["inflasi 3.2%", "pertumbuhan ekonomi 5.1%", "ekspor naik 8%"]
    actions = ["melarang", "mewajibkan", "menghapus", "membatasi"]
    events = ["seminar nasional", "konferensi internasional", "workshop"]
    projects = ["jembatan baru", "bandara internasional", "jalan tol"]
    policies = ["RUU Cipta Kerja", "kebijakan fiskal baru", "regulasi digital"]
    
    samples = []
    
    # Generate hoax samples
    for _ in range(num_samples // 2):
        pattern, label = random.choice(hoax_patterns)
        text = pattern.format(
            topic=random.choice(topics),
            claim=random.choice(claims),
            entity=random.choice(entities),
            location=random.choice(locations),
            item=random.choice(items),
            action=random.choice(actions),
            event=f"{random.choice(events)} tentang {random.choice(topics)}"
        )
        samples.append({"content": text, "label": label})
    
    # Generate valid samples
    for _ in range(num_samples - num_samples // 2):
        pattern, label = random.choice(valid_patterns)
        text = pattern.format(
            ministry=random.choice(ministries),
            topic=random.choice(topics),
            institution=random.choice(institutions),
            finding=random.choice(findings),
            program=random.choice(programs),
            statistic=random.choice(statistics),
            entity=random.choice(institutions),
            project=random.choice(projects),
            location=random.choice(locations),
            policy=random.choice(policies),
            event=random.choice(events)
        )
        samples.append({"content": text, "label": label})
    
    # Shuffle
    random.shuffle(samples)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['content', 'label'])
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"✓ Generated {len(samples)} synthetic samples")
    print(f"  - Hoax: {sum(1 for s in samples if s['label'] == 1)}")
    print(f"  - Valid: {sum(1 for s in samples if s['label'] == 0)}")
    print(f"  - Saved to: {output_path}")
    
    return output_path


def check_data_availability() -> Dict[str, bool]:
    """Check which datasets are available locally."""
    files = {
        "turnbackhoax.csv": os.path.join(DATA_DIR, "turnbackhoax.csv"),
        "turnbackhoax.json": os.path.join(DATA_DIR, "turnbackhoax.json"),
        "synthetic_data.csv": os.path.join(DATA_DIR, "synthetic_turnbackhoax.csv"),
        "sample_documents.json": os.path.join(DATA_DIR, "sample_documents.json")
    }
    
    return {name: os.path.exists(path) for name, path in files.items()}


def print_data_status():
    """Print status of available datasets."""
    status = check_data_availability()
    
    print("\n" + "=" * 50)
    print("📂 Dataset Availability")
    print("=" * 50)
    
    for name, available in status.items():
        icon = "✓" if available else "✗"
        print(f"  {icon} {name}")
    
    if not any(status.values()):
        print("\n⚠️  No datasets found!")
        print("Run this script with --generate to create synthetic data:")
        print("  python -m src.data.download_data --generate")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TurnBackHoax Dataset Helper")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic training data")
    parser.add_argument("--samples", type=int, default=2000, help="Number of synthetic samples")
    parser.add_argument("--status", action="store_true", help="Check data availability")
    
    args = parser.parse_args()
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if args.status:
        print_data_status()
    
    elif args.generate:
        output_path = os.path.join(DATA_DIR, "synthetic_turnbackhoax.csv")
        create_synthetic_turnbackhoax(output_path, args.samples)
        print("\n✓ Synthetic data generated successfully!")
        print("Now you can train the hoax detector:")
        print(f"  python -m src.hoax_detection.train_lora --data_path {output_path}")
    
    else:
        print_data_status()
        print("Options:")
        print("  --generate    Generate synthetic training data")
        print("  --status      Check data availability")
        print("  --samples N   Number of synthetic samples (default: 2000)")
