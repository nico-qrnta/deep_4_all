# Compte-rendu TP3 : Optimisation de l'Oracle du Donjon

## 1. Choix du Modèle (Linear -> RNN -> LSTM)
Au début, on a testé un modèle **linéaire**, mais c'était pas terrible (environ 56% d'accuracy) parce qu'il ne prend pas en compte l'ordre des événements. En passant au **RNN**, ça allait un peu mieux mais c'était instable sur les longues séquences. 
Finalement, le **LSTM** est clairement le meilleur : il arrive à "retenir" des infos importantes du début du donjon (comme l'amulette) jusqu'à la fin. On est monté rapidement au-dessus de 80%.

## 2. Mode Bidirectionnel
L'activation du mode **bidirectionnel** a été le plus gros boost. Le modèle lit la séquence dans les deux sens, ce qui lui permet de mieux comprendre le contexte global (savoir qu'un dragon arrive à la fin aide à comprendre l'utilité d'une potion au début). On a gagné quasiment 10% d'accuracy direct.

## 3. Hyperparamètres et Stabilité
On a affiné les réglages pour éviter que le modèle n'apprenne par coeur (overfitting) :
*   **Embed_dim (4)** et **Hidden_dim (8)** : On a gardé des petites valeurs pour que le modèle reste simple et généralise bien.
*   **Num_layers (3)** : Ça donne assez de profondeur pour les règles complexes.
*   **Dropout (0.3)** : Indispensable pour la stabilité.
*   **Use_scheduler** : Super utile pour ralentir l'apprentissage quand on stagne.

## 4. Stratégie d'Optimisation (Adam -> SGD)
On a mis en place un **switch d'optimiseur** à l'époque 10. 
*   On commence avec **Adam** pour descendre très vite vers une bonne solution.
*   On finit avec **SGD** (plus lent) pour "peaufiner" et stabiliser le modèle au fond du minimum. 
Ça permet d'avoir un modèle très précis sans les oscillations d'Adam à la fin.

**Commande finale utilisée :**
```bash
uv run python3 ./train_dungeon_logs.py --embed_dim 4 --hidden_dim 8 --num_layers 3 --dropout 0.3 --mode lstm --epoch 20 --bidirectional --early_stopping --optimizer adam --learning_rate 0.01 --use_scheduler --switch_epoch 10
```
Résultat : On atteint plus de 95% d'accuracy sur quasiment toutes les catégories de donjons. avec 1000 paramètres
