## 1. Choix du Modèle (Linear -> RNN -> LSTM -> GRU)
Au début, on a testé un modèle **linéaire**, mais c'était pas terrible (environ 56% d'accuracy). En passant au **RNN**, ça allait un peu mieux mais c'était instable. 
Le **LSTM** était bien plus performant, mais on n'arrivait pas à descendre sous la barre des 4000 paramètres tout en gardant une bonne précision.
Finalement, on a opté pour le **GRU** : c'est un excellent compromis. Il est plus léger que le LSTM mais bien plus "intelligent" qu'un RNN classique. Ça nous a permis de casser les records de compacité.

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

**Commande finale utilisée (GRU) :**
```bash
uv run python3 ./train_dungeon_logs.py --embed_dim 4 --hidden_dim 8 --num_layers 1 --mode gru --epoch 30 --bidirectional --early_stopping --optimizer adam --learning_rate 0.01 --use_scheduler --switch_epoch 23
```
Résultat : On atteint un record de **97.67% d'accuracy** avec seulement **800 paramètres**, là où on galérait à descendre sous les 4000 avec un LSTM.
