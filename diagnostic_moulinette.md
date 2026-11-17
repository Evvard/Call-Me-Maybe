# Diagnostic de la moulinette Call Me Maybe

Vérifications du 14 novembre 2026, à partir de la moulinette présente dans le projet.

## Cause reproduite avant correction

Le jeu public comporte 5 fonctions et 11 demandes. Le jeu privé comporte
6 fonctions et 11 demandes. La moulinette transforme les annotations Python
`int` en type JSON `integer`.

Deux définitions privées utilisent ce type :

| Fonction | Paramètre | Type attendu |
|---|---|---|
| `fn_is_even` | `n` | `integer` |
| `fn_calculate_compound_interest` | `years` | `integer` |

Le parseur de l'application n'acceptait que `number`, `string` et `boolean`.
Le lancement réel sur le jeu privé s'arrêtait donc avec un code de sortie 1 :

```text
Error: 1 validation error for FonctionDefinition
parameters.n.type
Input should be 'number', 'string' or 'boolean'
```

Aucun fichier de résultats n'était créé. Le chargement des définitions précède
celui de Qwen : cet échec survient avant toute génération.

Cela démontre une cause bloquante pour le jeu privé. La répartition exacte des
80 points obtenus lors de l'évaluation ne peut pas être reconstituée sans sa
grille ou son journal.

## Pourquoi un flottant ne suffit pas

Les fonctions de la moulinette exécutent notamment :

```python
assert isinstance(n, int)
assert isinstance(years, int)
```

`4` est un entier ; `4.0` est un flottant. La même valeur mathématique ne suffit
donc pas : le type Python après lecture du JSON compte aussi.

## Correction ciblée autorisée

- `src/parser.py` : accepter le type `integer`, conserver `int` dans l'union
  des valeurs, exiger `type(value) is int` pour ces paramètres.
- `src/fn_add_nb.py` : autoriser les préfixes d'entiers, exclure les points et
  exposants pour ce type, retourner un `int` et reconnaître sa fin de génération.
- `README.md` : décrire cette prise en charge.

Le SDK, `function_select.py` et les sources de la moulinette ne sont pas modifiés.
La correction ne contient aucun nom de fonction ni aucune réponse codée en dur
pour les exemples de la moulinette.

## Vérifications

Six tests temporaires ont d'abord reproduit les erreurs, puis sont passés après
correction : chargement du jeu privé, préfixes entiers, conversion conservant
le type `int`, rejet des conversions implicites, validation et sérialisation,
et maintien du comportement des anciens types.

Les vérifications antérieures des nombres, booléens, échappements, Unicode,
espaces, tokens contenant de la ponctuation et limite de génération passent aussi.

Flake8 sur `src` et mypy strict sur `src` passent. Les commandes complètes
`make lint` et `make lint-strict` échouent actuellement sur 52 diagnostics Flake8
dans le dossier externe `moulinette/`, nouvellement ajouté et non exclu des
contrôles. Le code de ce correcteur fourni n'a pas été retouché pour les masquer.

## Résultat public confirmé par la moulinette

La moulinette réelle note l'ancien fichier public à **10/11, soit 90,9 %**.
Elle exécute les fonctions demandées avec les arguments fournis et compare
le résultat obtenu à celui attendu.

Le remplacement des voyelles échoue avec `replacement = " *"` au lieu de `"*"` :

```text
Obtenu  : Pr *gr *mm *ng  *s f *n
Attendu : Pr*gr*mm*ng *s f*n
```

Ce défaut sémantique est distinct du refus des entiers. Le correctif des entiers
ne supprime pas arbitrairement les espaces dans les chaînes.

## Résultat privé après correction

Le lancement réel avec Qwen s'est terminé avec un code de sortie 0 et a produit
11 appels conformes au schéma, en **307,54 secondes**, soit **5 min 7,54 s**.
Cette mesure dépasse légèrement les cinq minutes demandées par le sujet.

La moulinette réelle attribue **9/11, soit 81,8 %** à ce fichier.

| Cas | Attendu | Produit | Résultat |
|---|---|---|---|
| Multiplications | Deux résultats corrects | Arguments corrects | 2/2 |
| Parité | `n=4` et `n=7`, entiers | `4` et `7`, types conservés | 2/2 |
| Intérêts composés | `years=23`, entier | Tous les arguments corrects | 1/1 |
| Requêtes SQL | Deux chaînes exactes | Chaînes correctes | 2/2 |
| Chemin Linux | `/home/user/data.json` | `home/user/data.json` | Échec : slash initial perdu. |
| Chemin Windows | Chemin du test fourni | Chaîne correcte | 1/1 |
| Premier modèle de texte | `Hello {user}'s profile!` | Chaîne correcte | 1/1 |
| Second modèle de texte | `Say "hello" to {name}` | `Say hello to {name}` | Échec : guillemets perdus. |

Les trois tests utilisant des arguments entiers passent désormais. Les deux
échecs restants concernent l'extraction exacte des chaînes ; ils ne sont pas
résolus par ce correctif limité aux entiers.

Attention : ce programme de correction affiche `PASSED` dès 70 % de réussite.
Son message `PASSED` à 81,8 % ne prouve donc pas le respect de l'objectif de
90 % du sujet.

## Refaire les tests avec les commandes de la moulinette

Depuis la racine du projet, générer le jeu privé dans un dossier réservé à ces
essais. La préparation recrée ses sous-dossiers `input` et `correction` :

```sh
uv run --project moulinette python -m moulinette prepare_exercises \
  --set private --output moulinette/data_private

uv run python -m src \
  --functions_definition moulinette/data_private/input/functions_definition.json \
  --input moulinette/data_private/input/function_calling_tests.json \
  --output data/output/private_calls.json

uv run --project moulinette python -m moulinette grade_student_answers \
  data/output/private_calls.json --set private
```

L'application reçoit seulement les définitions et les questions. Elle ne lit
ni les corrections ni le code des fonctions de la moulinette.
