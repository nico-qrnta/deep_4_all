import csv
import random
import os

def generate_cursed_data(n_samples=2000, output_path="data/train_cursed.csv"):
    """
    Génère des données synthétiques suivant les règles des "Terres Maudites"
    SANS dépendances externes (numpy/pandas).
    """
    random.seed(42)
    
    header = ['force', 'intelligence', 'agilite', 'chance', 'experience', 'niveau_quete', 'equipement', 'fatigue', 'survie']
    
    samples = []
    survival_count = 0
    arrogance_count = 0
    arrogance_survival = 0
    
    for i in range(n_samples):
        # Caractéristiques de base (0-100)
        force = random.uniform(0, 100)
        intelligence = random.uniform(0, 100)
        agilite = random.uniform(0, 100)
        chance = random.uniform(0, 100)
        experience = random.uniform(0, 20)
        niveau_quete = float(random.randint(1, 10))
        equipement = random.uniform(0, 100)
        fatigue = random.uniform(0, 100)
        
        # --- FORMULE DES TERRES MAUDITES ---
        # Intelligence 30%, Agilité 20%, Chance 20%, Équipement 15%, Expérience 5%
        # Force (<70) 10%
        # MALUS : Fatigue -10%, Difficulté -10%, ARROGANCE -15% (Force >70)
        
        score = (
            0.30 * intelligence +
            0.20 * agilite +
            0.20 * chance +
            0.15 * equipement +
            0.05 * (experience / 20 * 100)
        )
        
        # Règle de la Force et de l'Arrogance
        if force < 70:
            score += 0.10 * (force / 70 * 100)
        else:
            score -= 15
            arrogance_count += 1
            
        score -= 0.10 * fatigue
        score -= 0.10 * (niveau_quete * 10)
        
        # Seuil de survie équilibré
        survie = 1 if score > 30 else 0
        if survie:
            survival_count += 1
            if force >= 70:
                arrogance_survival += 1
        
        samples.append([force, intelligence, agilite, chance, experience, niveau_quete, equipement, fatigue, survie])
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(samples)
        
    print(f"✅ Généré {n_samples} échantillons dans {output_path}")
    print(f"📊 Taux de survie : {survival_count/n_samples:.2%}")
    print(f"💪 Cas d'arrogance (Force > 70) : {arrogance_count}")
    print(f"💀 Survie en cas d'arrogance : {arrogance_survival/arrogance_count if arrogance_count > 0 else 0:.2%}")

if __name__ == "__main__":
    generate_cursed_data()
