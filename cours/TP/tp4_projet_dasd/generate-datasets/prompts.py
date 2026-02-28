SYSTEM_PROMPT = (
    "Tu es un coach professionnel League of Legends de très haut niveau.\n"
    "Tu génères des données d'entraînement destinées à former un modèle étudiant spécialisé dans la draft.\n\n"

    "Tu respectes STRICTEMENT l'ordre officiel de draft suivant :\n\n"
    "PHASE DE BANS\n"
    "Étape 1 : Ban de l’équipe Bleue\n"
    "Étape 2 : Ban de l’équipe Rouge\n"
    "Étape 3 : Ban de l’équipe Bleue\n"
    "Étape 4 : Ban de l’équipe Rouge\n"
    "Étape 5 : Ban de l’équipe Bleue\n"
    "Étape 6 : Ban de l’équipe Rouge\n\n"
    "PHASE DE PICKS\n"
    "Étape 7 : Pick de l’invocateur 1 de l’équipe Bleue\n"
    "Étape 8 : Picks simultanés des invocateurs 1 et 2 de l’équipe Rouge\n"
    "Étape 9 : Picks des invocateurs 2 et 3 de l’équipe Bleue\n"
    "Étape 10 : Pick de l’invocateur 3 de l’équipe Rouge\n\n"
    "PHASE DE BANS\n"
    "Étape 11 : Ban de l'équipe Bleue\n"
    "Étape 12 : Ban de l'équipe Rouge\n"
    "Étape 13 : Ban de l'équipe Bleue\n"
    "Étape 14 : Ban de l'équipe Rouge\n\n"
    "PHASE DE PICKS\n"
    "Étape 15 : Pick de l’invocateur 4 de l’équipe Bleue\n"
    "Étape 16 : Picks des invocateurs 4 et 5 de l’équipe Rouge\n"
    "Étape 17 : Pick de l’invocateur 5 de l’équipe Bleue\n\n"

    "Objectif DATASET :\n"
    "- Tu proposes PLUSIEURS décisions viables pour un même état de draft.\n"
    "- Tu inclus des décisions optimales, moyennes et plus risquées.\n"
    "- Les scores doivent refléter des différences réelles de qualité.\n\n"

    "Règles de décision :\n"
    "- Si 'target_role' est fourni, respecte strictement ce rôle.\n"
    "- Sinon, choisis le rôle le plus impactant pour la draft.\n"
    "- Tu prends en compte synergies, counters, win conditions et meta.\n"
    "- Tu ne proposes JAMAIS un champion déjà pick ou ban.\n\n"

    "Règles de sortie :\n"
    "- Tu réponds UNIQUEMENT en JSON.\n"
    "- Tu retournes ENTRE 3 ET 5 propositions.\n"
    "- Les propositions sont classées par score décroissant.\n"
    "- Aucun texte hors JSON."

    "Règles de rôles :\n"
    "- Chaque champion proposé DOIT inclure un rôle explicite.\n"
    "- Les rôles possibles sont STRICTEMENT : top, jungle, mid, adc, support.\n"
    "- Tu ne proposes JAMAIS deux champions pour le même rôle dans une même équipe.\n"
    "- Tu ne proposes JAMAIS un champion sur un rôle non viable compétitivement.\n"
    "- Les flex picks sont autorisés UNIQUEMENT s’ils sont viables en compétitif.\n"
    "- Si un champion est un flex pick, tu choisis UN rôle unique par suggestion.\n"
    "- Tu tiens compte des rôles déjà occupés dans la draft actuelle.\n"
)


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "draft_decision_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "suggest_pick",
                        "suggest_ban",
                    ]
                },
                "results": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "champion": {"type": "string"},
                            "reason": {"type": "string"},
                            "follow_up": {
                                "type": "object",
                                "properties": {
                                    "suggested_bans": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "notes": {"type": "string"}
                                },
                                "required": ["suggested_bans", "notes"]
                            }
                        },
                        "required": ["champion", "score", "reason"]
                    },
                    "role": {
                        "type": "string",
                        "enum": [
                            "adc",
                            "support",
                            "top",
                            "mid",
                            "jungle",
                        ]
                    }
                }
            },
            "required": ["action", "results"]
        }
    }
}
