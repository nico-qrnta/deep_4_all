"""
Script d'entraînement : Oracle du Donjon (Séquences)

Ce script entraîne le modèle DungeonOracle sur le dataset des journaux de donjon.
Il contient volontairement des pratiques non optimales à améliorer.

Problèmes à corriger :
1. Pas de padding intelligent (pack_padded_sequence)
2. Pas d'augmentation de données
3. Learning rate fixe (pas de scheduler)
4. Modèle baseline utilise RNN au lieu de LSTM
5. Embedding dimension trop petite
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from baseline_model import DungeonOracle, count_parameters


# ============================================================================
# Dataset PyTorch pour séquences
# ============================================================================


class DungeonLogDataset(Dataset):
    """Dataset des journaux de donjon (séquences d'événements)."""

    def __init__(
            self,
            csv_path: str,
            vocab_path: str,
            max_length: int = 140,
            ):
        """
        Args:
            csv_path: Chemin vers le fichier CSV des logs
            vocab_path: Chemin vers le fichier JSON du vocabulaire
            max_length: Longueur max des séquences (truncate si dépassé)
        """
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length

        # Charger le vocabulaire
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        self.pad_idx = self.vocab.get("<PAD>", 0)
        self.unk_idx = self.vocab.get("<UNK>", 1)

        # Pré-tokeniser toutes les séquences
        self.sequences = []
        self.labels = []
        self.lengths = []

        for _, row in self.df.iterrows():
            # Parser la séquence "Entree -> Rat -> Sortie" en liste
            events = [e.strip() for e in row['sequence'].split(' -> ')]

            # Convertir en IDs
            token_ids = [self.vocab.get(e, self.unk_idx) for e in events]

            # Truncate si nécessaire
            if max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            self.sequences.append(torch.tensor(token_ids, dtype=torch.long))
            self.labels.append(row['survived'])
            self.lengths.append(len(token_ids))

        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.lengths = torch.tensor(self.lengths, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        length = self.lengths[idx]

        # Padding à max_length si spécifié
        if self.max_length is not None:
            padded_sequence = torch.full(
                (self.max_length,),
                self.pad_idx,
                dtype=torch.long
            )
            # Copier la séquence originale (truncated si nécessaire)
            original_length = len(sequence)
            seq_len = min(original_length, self.max_length)
            padded_sequence[:seq_len] = sequence[:seq_len]
            sequence = padded_sequence

        return sequence, self.labels[idx], length

    @property
    def vocab_size(self):
        return len(self.vocab)


# ============================================================================
# Boucle d'entraînement
# ============================================================================


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîne le modèle pour une epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels, lengths in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences, lengths).squeeze()

        # Loss et backward
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping (important pour RNN!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistiques
        total_loss += loss.item() * len(labels)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Évalue le modèle."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(sequences, lengths).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def evaluate_by_category(model, dataloader, device, df):
    """Évalue le modèle par catégorie de donjon."""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            outputs = model(sequences, lengths).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculer accuracy par catégorie
    results = {}
    for cat in df['category'].unique():
        mask = df['category'] == cat
        indices = mask[mask].index.tolist()

        if len(indices) > 0:
            cat_preds = [all_preds[i] for i in indices if i < len(all_preds)]
            cat_labels = [all_labels[i] for i in indices if i < len(all_labels)]

            if len(cat_preds) > 0:
                correct = sum(p == l for p, l in zip(cat_preds, cat_labels))
                results[cat] = {
                    'accuracy': correct / len(cat_preds),
                    'count':    len(cat_preds)
                    }

    return results


# ============================================================================
# Fonction principale
# ============================================================================


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Chemins
    data_dir = Path(__file__).parent / "data"
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    vocab_path = data_dir / "vocabulary_dungeon.json"
    train_path = data_dir / "train_dungeon.csv"
    val_path = data_dir / "val_dungeon.csv"

    val_df = pd.read_csv(val_path)

    # Charger les datasets
    print("Chargement des données...")
    train_dataset = DungeonLogDataset(
            str(train_path),
            str(vocab_path)
            )
    val_dataset = DungeonLogDataset(
            str(val_path),
            str(vocab_path)
            )

    # DataLoaders
    train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
            )
    val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
            )

    print(f"Train: {len(train_dataset)} séquences")
    print(f"Val: {len(val_dataset)} séquences")
    print(f"Vocabulaire: {train_dataset.vocab_size} tokens")

    # Statistiques des longueurs
    train_lengths = train_dataset.lengths.numpy()
    print(
            f"Longueur des séquences: min={train_lengths.min()}, "
            f"max={train_lengths.max()}, mean={train_lengths.mean():.1f}"
            )

    # Modèle
    print("\nCréation du modèle...")
    model = DungeonOracle(
            vocab_size=train_dataset.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            mode=args.mode,
            max_length=train_dataset.max_length,
            bidirectional=args.bidirectional,
            padding_idx=train_dataset.pad_idx,
            )
    model = model.to(device)

    print(f"Architecture: {args.mode}")
    print(f"Bidirectionnel: {args.bidirectional}")
    print(f"Paramètres: {count_parameters(model):,}")

    # Loss et optimiseur
    criterion = nn.BCEWithLogitsLoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
                )
    else:
        optimizer = optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=args.weight_decay
                )

    # Scheduler (optionnel)
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
                )

    print(f"Optimiseur: {args.optimizer.upper()}, LR: {args.learning_rate}")

    # Historique
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc': []
        }

    # Entraînement
    print("\n" + "=" * 60)
    print("Début de l'entraînement")
    print("=" * 60)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        if epoch == args.switch_epoch:
            print("🔄 Switching from ADAM to SGD for fine-tuning...")
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate / 10, momentum=0.9)

        # Train
        train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
                )

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler
        if scheduler:
            scheduler.step(val_acc)

        # Historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Affichage
        print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
                )

        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, checkpoint_dir / "best_dungeon_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\nEarly stopping après {epoch + 1} epochs")
            break

    print("\n" + "=" * 60)
    print(f"Meilleure accuracy validation: {best_val_acc:.2%}")
    print(f"Modèle sauvegardé: {checkpoint_dir / 'best_dungeon_model.pt'}")
    print("=" * 60)

    # Sauvegarder l'historique
    with open(checkpoint_dir / "dungeon_history.json", 'w') as f:
        json.dump(history, f, indent=4)

    # Évaluation par catégorie (piège pédagogique!)
    print("\n" + "-" * 60)
    print("Analyse par catégorie de donjon")
    print("-" * 60)

    cat_results = evaluate_by_category(model, val_loader, device, val_df)

    for cat, stats in sorted(cat_results.items()):
        print(f"  {cat:30s}: {stats['accuracy']:.2%} ({stats['count']} ex.)")

    # Alertes pédagogiques
    print("\n" + "!" * 60)
    print("POINTS D'ATTENTION:")

    # Gap train/val
    gap = history['train_acc'][-1] - history['val_acc'][-1]
    if gap > 0.10:
        print(f"  - OVERFITTING: Gap train-val de {gap:.2%}")
        print("    -> Augmentez dropout, reduisez hidden_dim, ou ajoutez regularisation")

    print("!" * 60)

    # Plot
    if args.plot:
        plot_history(history, checkpoint_dir / "dungeon_training_curves.png")


def plot_history(history, save_path):
    """Affiche les courbes d'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss au cours de l\'entraînement')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy au cours de l\'entraînement')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nCourbes sauvegardées: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Entraînement de l'Oracle du Donjon (Séquences)"
            )
    # Modèle
    parser.add_argument(
            '--embed_dim', type=int, default=258,
            help='Dimension des embeddings'
            )
    parser.add_argument(
            '--hidden_dim', type=int, default=258,
            help='Dimension de l\'état caché RNN/LSTM'
            )
    parser.add_argument(
            '--num_layers', type=int, default=1,
            help='Nombre de couches RNN/LSTM'
            )
    parser.add_argument(
            '--dropout', type=float, default=0.0,
            help='Dropout entre les couches RNN'
            )
    parser.add_argument(
            '--mode',
            type=str,
            default='linear',
            choices=['linear', 'rnn', 'lstm', 'gru'],
            help='Architecture du modèle (default: %(default)s)')
    parser.add_argument(
            '--bidirectional', action='store_true', default=False,
            help='RNN/LSTM bidirectionnel'
            )

    # Entraînement
    parser.add_argument(
            '--epochs', type=int, default=6,
            help='Nombre d\'epochs'
            )
    parser.add_argument(
            '--batch_size', type=int, default=32,
            help='Taille du batch'
            )
    parser.add_argument(
            '--learning_rate', type=float, default=0.1,
            help='Learning rate'
            )
    parser.add_argument(
            '--optimizer', type=str, default='sgd',
            choices=['adam', 'sgd'],
            help='Optimiseur'
            )
    parser.add_argument(
            '--weight_decay', type=float, default=0.0,
            help='Weight decay (L2 regularization)'
            )
    parser.add_argument(
            '--use_scheduler', action='store_true', default=False,
            help='Utiliser un learning rate scheduler'
            )
    parser.add_argument(
            '--switch_epoch', type=int, default=-1,
            help='Epoch à laquelle basculer de ADAM à SGD (-1 pour désactiver)'
            )

    # Early stopping
    parser.add_argument(
            '--early_stopping', action='store_true', default=False,
            help='Activer early stopping'
            )
    parser.add_argument(
            '--patience', type=int, default=10,
            help='Patience pour early stopping'
            )

    # Autres
    parser.add_argument(
            '--plot', action='store_true', default=True,
            help='Afficher les courbes'
            )

    args = parser.parse_args()

    # Afficher la configuration
    print("=" * 60)
    print("ORACLE DU DONJON - Entraînement")
    print("=" * 60)
    print("\nConfiguration:")
    print("-" * 40)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 40)

    main(args)
