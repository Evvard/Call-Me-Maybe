# Call Me Maybe — comprendre le projet et préparer la soutenance

Guide en français fondé sur le code présent le **17 novembre 2026**.
Les explications suivent l'implémentation actuelle, y compris les entiers et le
contexte raccourci pour générer les arguments. Chaque fonction possède un résumé
rapide, puis une explication détaillée. Le SDK est expliqué sans être modifié.

## Sommaire et ordre de lecture conseillé

1. [Le but du projet et les mots à connaître](#1-le-but-du-projet)
2. [Les fichiers et leur rôle](#2-les-fichiers)
3. [L'ordre réel d'exécution](#3-ordre-execution)
4. [Les modèles et fonctions de parser.py](#4-parser)
5. [Le choix de fonction dans function_select.py](#5-selection)
6. [La génération des arguments dans fn_add_nb.py](#6-generation)
7. [Le lancement et la sauvegarde dans __main__.py](#7-main)
8. [Toutes les méthodes du SDK fourni](#8-sdk)
9. [Un exemple parcouru de bout en bout](#9-exemple)
10. [uv, Makefile et vérifications](#10-outils)
11. [Les changements utiles à comprendre et les limites actuelles](#11-limites)
12. [Révision et questions de soutenance](#12-revision)

**Si tu as peu de temps :** lis d'abord les parties 1, 3 et 9, puis tous les
« Résumé rapide ». Approfondis ensuite `select_valid_function`, `choose_token`,
`decode_parameter` et `validate_call`. Termine par les questions de la partie 12.
Le document est un support d'explication : lis le fichier Python correspondant
à côté pour reconnaître les variables décrites.

<a id="1-le-but-du-projet"></a>

## 1. Le but du projet

Le programme reçoit une demande en langage naturel et une liste de descriptions
de fonctions. Il doit produire un appel structuré : **quelle fonction choisir,
avec quels arguments**, tout en conservant la demande d'origine.

Exemple pédagogique :

```text
Demande : Add 2 and 3.
Fonction disponible : fn_add_numbers(a: number, b: number)
```

Une sortie possible est :

```json
{
  "prompt": "Add 2 and 3.",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

**Le programme ne calcule pas la somme.** Il prépare les informations permettant
à un autre programme d'appeler `fn_add_numbers`. Pour une inversion de chaîne,
il faut transmettre la chaîne d'origine comme argument, pas la chaîne déjà
inversée. Le champ `returns` décrit un résultat éventuel ; il ne provoque pas
l'exécution de la fonction.

Deux responsabilités se complètent :

- **Le LLM** interprète la demande et propose les valeurs, grâce à ses scores.
- **Le code Python** impose les noms disponibles, les clés des arguments et les
  formes JSON autorisées, puis valide et sauvegarde la réponse.

### Les mots indispensables

| Mot | Sens concret dans le projet |
|---|---|
| Prompt | Texte de la demande de l'utilisateur. |
| Schéma | Description des champs et types attendus. |
| LLM | Modèle de langage ; ici, par défaut, Qwen/Qwen3-0.6B. |
| Token | Morceau de texte identifié par un entier dans le vocabulaire. |
| Tokenizer | Outil fourni par le SDK pour convertir texte et identifiants. |
| Contexte | Tout le texte déjà donné au modèle pour prédire la suite. |
| Logit | Score brut associé à un token possible pour la prochaine position. |
| Préfixe | Début de valeur qui peut encore être complété. |
| Scalaire | Valeur simple : chaîne, nombre, entier ou booléen ici. |
| Sérialisation | Conversion des objets Python en texte JSON pour le fichier. |
| Décodage contraint | Choisir les tokens en interdisant ceux qui casseraient les règles. |

Un token ne correspond pas forcément à une lettre ou un mot entier. Il peut
contenir du texte et de la ponctuation ensemble, par exemple une fin de chaîne
suivie d'une virgule. C'est une difficulté centrale de ce projet.

### Trois questions différentes pour juger la sortie

1. **Le JSON est-il lisible ?** Les guillemets et séparateurs sont corrects.
2. **Le schéma est-il respecté ?** Bonne fonction disponible, bonnes clés,
   bons types, nombres finis.
3. **La demande est-elle correctement comprise ?** Les valeurs expriment
   réellement ce que demandait l'utilisateur.

Les deux premières ne prouvent pas la troisième. Une chaîne vide est une chaîne
JSON valide, mais peut être une mauvaise réponse à une demande précise.

<a id="2-les-fichiers"></a>

## 2. Les fichiers et leur rôle

```text
call_me/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── parser.py
│   ├── function_select.py
│   └── fn_add_nb.py
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/
│       └── function_calls.json       ← créé lors d'une exécution réussie
├── llm_sdk/
│   ├── llm_sdk/__init__.py
│   ├── pyproject.toml
│   └── uv.lock
├── makefile
├── pyproject.toml
├── uv.lock
├── .flake8
├── .gitignore
├── README.md
└── doc.md
```

| Fichier | Responsabilité |
|---|---|
| `src/__init__.py` | Fichier vide qui identifie le paquet Python `src`. Aucune fonction à exécuter. |
| `src/__main__.py` | Point d'entrée : organise la lecture, le modèle, la génération et l'écriture. |
| `src/parser.py` | Définit les modèles Pydantic et vérifie les données d'entrée et de sortie. |
| `src/function_select.py` | Choisit un nom parmi les définitions disponibles avec les scores du modèle. |
| `src/fn_add_nb.py` | Génère tous les arguments scalaires ; son nom historique ne limite pas son rôle aux additions. |
| `data/input/functions_definition.json` | Décrit les fonctions disponibles, sans contenir leur implémentation Python. |
| `data/input/function_calling_tests.json` | Contient une liste de demandes, chacune sous la forme `{"prompt": "..."}`. |
| `data/output/function_calls.json` | Liste finale des appels ; son chemin peut être changé en argument. |
| `llm_sdk/llm_sdk/__init__.py` | SDK fourni : charge le modèle, encode, décode et renvoie les logits. |
| `llm_sdk/pyproject.toml` | Décrit les dépendances propres au SDK : torch, transformers et huggingface-hub. |
| `llm_sdk/uv.lock` | Verrouillage livré avec le sous-projet SDK. Le lancement à la racine utilise le projet racine. |
| `pyproject.toml` | Métadonnées, dépendances de l'application et configuration mypy/uv. |
| `uv.lock` | Versions exactes résolues des dépendances de l'environnement racine. |
| `makefile` | Raccourcis d'installation, lancement, débogage, nettoyage et lint. |
| `.flake8` | Style Python et dossiers exclus du contrôle. |
| `.gitignore` | Fichiers générés ou locaux à ne pas suivre dans Git. |
| `README.md` | Présentation du projet en anglais, installation et utilisation. |
| `doc.md` | Ce guide pédagogique en français. |

`.venv/` contient l'environnement Python local. `__pycache__/` et
`.mypy_cache/` sont des dossiers générés par les outils, pas de la logique métier.
Ils ne sont pas des fichiers source à expliquer fonction par fonction.

Le dossier externe `moulinette/`, lorsqu'il est présent, sert à l'évaluation.
L'application ne l'importe pas et ne lit pas ses corrections. Il peut être
retiré sans enlever une partie du programme. Un éventuel
`diagnostic_moulinette.md` est un compte rendu de diagnostic, pas du code appelé
par l'application ; les résultats plus anciens doivent être distingués des
mesures du 17 novembre rappelées en partie 11.

<a id="3-ordre-execution"></a>

## 3. L'ordre réel d'exécution

### Déclarer une fonction ne l'exécute pas

Quand Python importe un fichier, il exécute ses instructions au niveau du module
et crée les définitions de classes et de fonctions. Il n'exécute pas chaque
corps de fonction dans l'ordre des lignes du fichier.

Par exemple, `write_results` est défini avant `main`, mais son écriture du JSON
se produit à la fin du traitement. Les modèles Pydantic sont définis à l'import,
mais les données ne sont validées que lorsqu'on leur transmet des valeurs.

### Parcours d'un lancement normal

```text
make run
└── uv run python -m src
    ├── Python charge le paquet src et src/__main__.py
    ├── les imports chargent les modules et leurs définitions
    └── le bloc if __name__ == '__main__' appelle main()
        ├── parse_args()
        ├── vérification des chemins : ne pas écraser une entrée
        ├── validation_function()
        │   ├── load_json_list()
        │   └── FonctionDefinition.model_validate() pour chaque définition
        ├── construction du dictionnaire functions_map
        ├── load_prompts()
        │   ├── load_json_list()
        │   └── PrompTest.model_validate() pour chaque demande
        ├── si aucune demande : write_results(args.output, []), puis retour
        └── sinon :
            ├── Small_LLM_Model.__init__() : un chargement pour tout le lot
            ├── get_path_to_vocab_file(), puis lecture de vocab.json
            ├── build_vocabulary()
            │   └── model.decode() pour les tokens retenus
            ├── pour chaque demande, dans l'ordre du fichier :
            │   ├── select_valid_function()
            │   │   ├── model.encode() du contexte et des noms possibles
            │   │   └── choose_token() autant que nécessaire
            │   ├── reconstruction d'un contexte pour la seule fonction choisie
            │   ├── model.encode() de ce contexte et de la transition JSON
            │   ├── extract_all_parameters()
            │   │   └── decode_parameter() pour chaque argument
            │   │       ├── valid_prefix() / string_completion()
            │   │       ├── complete_number() selon le type
            │   │       ├── choose_token()
            │   │       └── parse_scalar(), puis ajout du suffixe au contexte
            │   ├── validate_call()
            │   ├── FunctionCallResult.model_dump()
            │   └── ajout du résultat à la liste en mémoire
            └── write_results() : écriture du lot complet
```

L'écriture n'a lieu qu'une fois pour un lot normal, après toutes les demandes.

### Les répétitions à retenir

| Fréquence | Travail effectué |
|---|---|
| Une fois au lancement | Lecture des options, définitions et demandes. |
| Une fois par lot non vide | Chargement du modèle et préparation du vocabulaire. |
| Une fois par demande | Choix de fonction, nouveau contexte, génération et validation. |
| Une fois par argument | Appel à `decode_parameter`. |
| Plusieurs fois par argument | Analyse des tokens candidats et choix du prochain token. |
| Une fois après le lot réussi | Sauvegarde du tableau JSON final. |

Lorsqu'il reste plusieurs tokens autorisés, `choose_token` demande des logits
au SDK. S'il ne reste qu'un seul identifiant autorisé, il l'utilise directement.

<a id="4-parser"></a>

## 4. `src/parser.py` : modèles et validation

### Les imports et les alias de types

- `json` lit les fichiers JSON et transforme leur contenu en objets Python.
- `math` sert ici à refuser les nombres infinis et `NaN`.
- `Path` représente un chemin de fichier et fournit `open`, entre autres.
- `Any`, `Literal`, `Union` décrivent les types attendus.
- `BaseModel`, `ConfigDict`, `Field` viennent de Pydantic, qui valide les données
  réellement fournies à l'exécution.

```python
ParameterValue = Union[str, float, int, bool]
ParameterType = Literal['number', 'integer', 'string', 'boolean']
```

`ParameterType` décrit une **étiquette dans le schéma**, par exemple `'integer'`.
`ParameterValue` décrit une **valeur d'argument**, par exemple `23`.
Ce ne sont pas deux façons d'écrire la même chose.

Une annotation Python aide mypy et le lecteur ; à elle seule, elle ne vérifie
pas toutes les valeurs à l'exécution. Pydantic et les conditions de validation
assurent cette vérification réelle.

### `ConfigDict(extra='forbid', strict=True)`

`model_config` est un attribut de configuration reconnu par Pydantic.
`ConfigDict` fournit le dictionnaire d'options utilisé pour ce modèle.

`extra='forbid'` refuse les champs non déclarés. Un objet `PrompTest` ne peut
donc pas contenir un champ supplémentaire `answer`.
`strict=True` évite les conversions permissives dans les champs du modèle :
un nombre ne doit pas être accepté comme texte pour `prompt`.

Attention à la portée : le champ `parameters` de `FunctionCallResult` est un
dictionnaire dont les valeurs peuvent avoir plusieurs types. Sa configuration
ne lui apprend pas que **cette fonction précise** exige uniquement `a` et `b`.
C'est `validate_call` qui fait cette vérification par rapport à la définition.

### Modèle `InputDetail`

**Résumé rapide**

Il décrit le type d'un argument, par exemple `{"type": "number"}`.
Il refuse les étiquettes inconnues et les champs supplémentaires.

**Explication détaillée**

Le modèle hérite de `BaseModel` et possède un champ obligatoire :
`type: ParameterType`. La définition `{"type": "integer"}` est donc autorisée ;
`{"type": "array"}` ne l'est pas dans cette implémentation.

Dans `parameters: {"age": {"type": "integer"}}`, le nom `age` est une clé du
dictionnaire de paramètres ; le modèle `InputDetail` décrit seulement la partie
`{"type": "integer"}`. Il n'y a pas de valeur par défaut ni de champ optionnel
déclaré : les arguments présents dans la définition sont traités comme requis.

### Modèle `OutputDetail`

**Résumé rapide**

Il décrit le type de retour annoncé par une fonction.
Il réutilise les mêmes champs et règles que `InputDetail` par héritage.

**Explication détaillée**

La classe ne redéfinit aucun champ : `class OutputDetail(InputDetail)` suffit
pour hériter de `type` et de la configuration. Le modèle est séparé pour donner
un sens clair au champ `returns` de la définition.
Il ne stocke pas un résultat calculé : aucune fonction métier n'est exécutée
par ce projet. Le type annoncé sert à décrire la fonction au modèle de langage.

### Modèle `FonctionDefinition`

**Résumé rapide**

Il représente une fonction disponible : nom, description, arguments et retour.
Il valide aussi les modèles imbriqués qui décrivent les types des arguments.

**Explication détaillée**

Les champs sont `name`, `description`, `parameters` et `returns`.
`Field(min_length=1)` interdit un nom vide ; il n'impose pas une convention de
nommage particulière. La description doit être une chaîne, mais aucune longueur
minimale n'est imposée à celle-ci.

`parameters: dict[str, InputDetail]` signifie que chaque nom d'argument pointe
vers une description validée. `returns: OutputDetail` valide la description du
retour. Avec `model_validate`, Pydantic construit ces objets imbriqués à partir
des dictionnaires lus dans le JSON.

Après validation, `definition.name` donne le nom et
`definition.parameters['a'].type` donne le type de `a`.
`model_dump()` effectue l'opération inverse utile ici : obtenir un dictionnaire
Python représentant le modèle, pour l'envoyer au LLM ou le sérialiser.

### Modèle `PrompTest`

**Résumé rapide**

Il conserve une demande utilisateur dans son champ `prompt`, obligatoirement textuel.
Il interdit les champs supplémentaires et ne réécrit pas la demande.

**Explication détaillée**

Un objet d'entrée attendu est `{"prompt": "Say hello to Alice."}`.
Un objet sans `prompt` ou avec `prompt: 42` est refusé.
Une chaîne vide reste autorisée par ce modèle, puisqu'aucune longueur minimale
n'est imposée. Le modèle vérifie la forme des données ; il ne décide pas si
le texte contient une demande sensée ou réalisable.

La chaîne d'origine est réutilisée dans la sortie. Elle n'est pas remplacée par
le contexte système, ni par une reformulation du modèle.

### Modèle `FunctionCallResult`

**Résumé rapide**

Il rassemble les trois champs finaux : `prompt`, `name` et `parameters`.
Il est construit après la validation des arguments propres à la fonction choisie.

**Explication détaillée**

Les deux premiers champs sont des chaînes. Le troisième est un dictionnaire de
valeurs scalaires : `str`, `float`, `int` ou `bool`.
L'ajout explicite de `int` permet notamment de conserver un entier comme entier
dans l'objet final, puis dans le JSON, lorsque le schéma exige `integer`.

Ce modèle ne contient ni champ `result`, ni champ `returns` : la sortie attendue
décrit un appel, pas le résultat de son exécution. `main` appelle ensuite
`validated_output.model_dump()` pour ajouter un dictionnaire à la liste finale.

### Fonction `load_json_list(path_json) -> list[Any]`

**Résumé rapide**

Elle lit un fichier JSON UTF-8 et exige un tableau à sa racine.
Elle ajoute le chemin du fichier au message si la lecture ou le décodage échoue.

**Explication détaillée**

**Appelée par :** `validation_function` et `load_prompts`, au début du lancement.
**Entrée :** une chaîne de chemin ou un objet `Path`.
**Sortie :** une liste Python dont les éléments ne sont pas encore validés.

1. `Path(path_json)` uniformise la représentation du chemin.
2. `with path.open(encoding='utf-8') as stream` ouvre le fichier et le referme
   automatiquement à la sortie du bloc, même en cas d'erreur.
3. `json.load(stream)` transforme le texte JSON en objets Python.
4. Les erreurs de fichier, d'encodage ou de JSON deviennent une `ValueError`
   avec le message `Cannot read ...`. `raise ... from error` conserve la cause.
5. Une racine différente d'une liste provoque une autre `ValueError`.

`[]` et `[{"prompt": "Hello"}]` sont des tableaux JSON.
`{"prompt": "Hello"}` est un objet JSON : il est refusé comme racine.
`Any` indique qu'à ce stade les éléments peuvent être de n'importe quelle forme ;
les modèles Pydantic font le tri ensuite.

### Fonction `validation_function(path_json) -> list[FonctionDefinition]`

**Résumé rapide**

Elle charge la liste des fonctions et valide chaque définition avec Pydantic.
Elle refuse une liste vide et des fonctions portant le même nom.

**Explication détaillée**

**Appelée par :** `main`, avant de charger les demandes puis le modèle.
**Entrée :** chemin des définitions. **Sortie :** liste de modèles validés.

La compréhension de liste appelle `load_json_list`, puis
`FonctionDefinition.model_validate(item)` sur chaque élément. Une erreur dans
une seule définition interrompt ce chargement : le programme ne l'ignore pas.

La liste ne doit pas être vide, car il faut disposer d'au moins une fonction à
sélectionner. Puis la fonction construit la liste des noms. Un `set` retire les
doublons ; si la longueur du `set` est différente de celle de la liste, un nom
était présent plusieurs fois. L'erreur évite un choix ambigu et un écrasement
silencieux lors de la construction de `functions_map` dans `main`.

### Fonction `load_prompts(path_json) -> list[PrompTest]`

**Résumé rapide**

Elle charge le tableau des demandes et valide chaque objet `prompt`.
Elle préserve l'ordre d'entrée et autorise une liste vide.

**Explication détaillée**

**Appelée par :** `main`, après validation des fonctions.
Elle utilise la même lecture JSON que les définitions, puis transforme chaque
élément en `PrompTest` avec `model_validate`.

Contrairement à la liste de fonctions, la liste de demandes peut être vide.
Dans ce cas, `main` écrit `[]` puis termine sans instancier le modèle : il n'y a
aucune génération utile à effectuer. L'ordre de la compréhension de liste est
celui du tableau d'origine ; cet ordre est conservé jusqu'au fichier de sortie.

### Fonction `validate_call(prompt, definition, parameters) -> FunctionCallResult`

**Résumé rapide**

Elle contrôle les clés et les types des arguments face à la fonction choisie.
Elle refuse les nombres non finis, puis construit l'objet de sortie validé.

**Explication détaillée**

**Appelée par :** `main`, après `extract_all_parameters`, pour chaque demande.
Elle reçoit la demande originale, un modèle `FonctionDefinition` et le
dictionnaire des valeurs générées.

La première condition compare `parameters.keys()` et
`definition.parameters.keys()`. Cette égalité vérifie l'ensemble des clés,
sans exiger un ordre particulier. Un argument manquant **ou** supplémentaire
provoque une erreur.

Puis elle parcourt les paramètres de la définition :

| Type du schéma | Condition sur la valeur Python |
|---|---|
| `number` | `int` ou `float`, mais pas `bool`, et `math.isfinite(value)` vrai. |
| `integer` | `type(value) is int` : entier exact. |
| `boolean` | `type(value) is bool`. |
| `string` | `type(value) is str`. |

Pourquoi refuser explicitement les booléens parmi les nombres ? En Python,
`isinstance(True, int)` vaut `True`, car `bool` hérite de `int`. Le schéma JSON
distingue pourtant une vérité logique d'un nombre.

`math.isfinite` refuse `NaN`, `+inf` et `-inf`. Une erreur indique la fonction,
le paramètre, le type attendu et la valeur reçue. Si tout est correct,
`FunctionCallResult` est construit avec une copie du dictionnaire.

Cette fonction ne relit pas la phrase pour vérifier le sens des valeurs.
`{"a": 2.0, "b": 99.0}` peut respecter le schéma d'une addition tout en étant
faux pour « Add 2 and 3 ». Cette distinction explique les erreurs sémantiques
qui restent possibles malgré un JSON conforme.

<a id="5-selection"></a>

## 5. `src/function_select.py` : choisir la fonction

### Fonction `select_valid_function(prompt, functions, model) -> tuple[str, list[int]]`

**Résumé rapide**

Elle laisse le LLM choisir parmi les noms de fonctions réellement disponibles.
À chaque token, elle interdit les continuations qui ne mènent à aucun de ces noms.
Elle renvoie le nom retenu et les identifiants du contexte construit.

**Explication détaillée**

**Appelée par :** `main`, une fois par demande.
**Entrées :** le texte utilisateur, la liste des définitions validées et le SDK.
**Sortie :** un tuple décomposable en `chosen_name, current_ids`.

**1. Préparer les consignes.** Si la liste de fonctions est vide, la fonction
lève une erreur. Chaque définition est transformée avec `model_dump`.
Les consignes demandent un objet JSON avec nom et paramètres, la copie des
arguments sans exécuter la fonction et l'inférence des regex si nécessaire.
Toutes les définitions disponibles sont incluses à cette étape.

**2. Construire le contexte de conversation.** Les marqueurs
`<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant` et
`<|im_end|>` séparent consignes, demande et début de réponse selon le format
utilisé pour Qwen. La réponse commence déjà par `{"name": `.
`model.encode(context).tolist()[0]` transforme ce contexte en liste d'identifiants.

**3. Construire les branches autorisées.** Pour chaque fonction,
`json.dumps(function.name)` produit son nom avec les guillemets JSON. Le SDK
encode ce texte. `branches` associe donc chaque nom à une séquence de tokens.
Ces identifiants proviennent du tokenizer fourni, pas d'une table inventée.

**4. Garder les branches compatibles.** `generated` contient seulement les
tokens du nom déjà sélectionnés. À chaque tour, `active` garde les noms dont
le début correspond exactement à `generated`.

Exemple symbolique, dont les identifiants sont inventés pour l'explication :

```text
Nom A : [10, 20, 30]
Nom B : [10, 21, 30]
Début : []       → prochain identifiant autorisé : 10
Début : [10]     → prochains identifiants autorisés : 20 ou 21
Choix du LLM : 21
Début : [10, 21] → prochain identifiant autorisé : 30
Séquence finie   → nom B
```

**5. Choisir puis terminer.** `choose_token` sélectionne le meilleur score
autorisé. L'identifiant est ajouté au contexte et à `generated`. Une séquence
complète provoque le retour de son nom. Les guillemets font partie des
séquences : ils participent à distinguer un nom complet d'un autre nom qui
partagerait son début textuel.

La fonction ne contient aucune règle du genre « si la phrase contient `add`,
alors choisir l'addition ». Le sens est traité par le modèle ; le code ne fait
que limiter le domaine de ses choix.

**Après son retour :** `main` garde le nom choisi, mais reconstruit le contexte
des arguments avec uniquement la définition retenue. La liste `current_ids`
retournée ici n'est donc pas réutilisée telle quelle pour les arguments dans
la version actuelle. Cela évite de renvoyer toutes les autres définitions au
modèle à chaque token d'argument.

<a id="6-generation"></a>

## 6. `src/fn_add_nb.py` : générer les arguments

Ce fichier est générique malgré son nom. Il gère les valeurs `number`,
`integer`, `string` et `boolean`, quel que soit le nom de la fonction choisie.
Il ne contient pas les implémentations de `fn_add_numbers`, `fn_greet`, etc.

### Les variables à reconnaître

| Variable | Contenu |
|---|---|
| `vocab` | Dictionnaire lu dans le vocabulaire : représentation du token → identifiant. |
| `pieces` ou `tokens` | Identifiant → morceau de texte décodé par le SDK. |
| `current_ids` | Liste des identifiants du contexte donné au modèle. |
| `text` | Texte JSON en cours de génération pour un seul argument. |
| `kind` | Type attendu : `'string'`, `'number'`, `'integer'` ou `'boolean'`. |
| `allowed` | Liste des identifiants encore autorisés au tour courant. |
| `suffix` | Structure attendue après la valeur : prochaine clé ou fin d'objet. |
| `remaining` | Partie du suffixe qu'il reste à ajouter au contexte. |
| `endings` | Tokens qui terminent une chaîne, avec le littéral et le suffixe restant. |

### Fonction `build_vocabulary(vocab, model) -> dict[int, str]`

**Résumé rapide**

Elle prépare une correspondance entre identifiants et texte des tokens utilisables.
Elle utilise le décodage du SDK et écarte les marqueurs spéciaux et décodages incomplets.

**Explication détaillée**

**Appelée par :** `main`, une fois après lecture du vocabulaire.
**Entrées :** le dictionnaire de vocabulaire et le SDK chargé.
**Sortie :** `pieces`, dictionnaire dont la clé est un entier et la valeur du texte.

Pour chaque paire `spelling, token_id`, la fonction ignore d'abord les
représentations commençant par `<|`, qui correspondent ici aux marqueurs
spéciaux. Elle appelle ensuite `model.decode([token_id])` pour obtenir le texte
associé au token. Elle conserve seulement les résultats non vides et sans
caractère de remplacement Unicode `\ufffd`.

Pourquoi décoder plutôt que réutiliser directement `spelling` ? La forme stockée
dans le vocabulaire peut représenter les espaces et les octets d'une manière
interne au tokenizer. La grammaire doit examiner le texte décodé. Le SDK reste
responsable de cette conversion.

Cette table est préparée une fois pour éviter de redécoder chaque identifiant
pendant chaque test de préfixe. Ce n'est pas un cache de réponses ni un cache
d'inférence du modèle. Certains tokens ne représentant qu'un fragment Unicode
isolé sont écartés ; il ne faut donc pas présenter ce filtrage comme une prise
en charge sans restriction de tous les découpages Unicode possibles.

### Fonction `complete_number(text, kind='number') -> bool`

**Résumé rapide**

Elle vérifie si un texte est déjà un nombre JSON complet du type attendu.
Elle distingue un entier exact d'un nombre général et refuse les valeurs non finies.

**Explication détaillée**

**Appelée par :** `valid_prefix` et `decode_parameter` pour les types numériques.
Elle commence par `json.loads(text)` : c'est le parseur JSON Python qui vérifie
la syntaxe, au lieu de supposer qu'une suite de chiffres et de points suffit.

Pour `integer`, le résultat doit avoir exactement le type Python `int`.
Pour `number`, il doit être de type `int` ou `float` et fini.
Les exceptions `ValueError` et `OverflowError` donnent `False` plutôt que
d'interrompre l'exploration des tokens candidats.

| Texte examiné | `number` | `integer` |
|---|---|---|
| `23` | vrai | vrai |
| `-5` | vrai | vrai |
| `23.0` | vrai | faux |
| `2e3` | vrai | faux |
| `01` | faux | faux |
| `1.` | faux, pas encore complet | faux |
| `true` | faux | faux |
| `1e9999` | faux, devient infini | faux |

`23.0` représente mathématiquement un entier, mais `json.loads` en fait un
`float`. Le schéma `integer` impose ici un vrai `int`. C'est utile quand une
fonction destinataire emploie une opération qui exige un entier.

### Fonction `string_prefix(text) -> bool`

**Résumé rapide**

Elle vérifie un début de chaîne JSON encore ouvert, y compris ses échappements.
Elle accepte les débuts incomplets qui pourront devenir valides avec les tokens suivants.

**Explication détaillée**

**Appelée par :** `valid_prefix` lorsque le type attendu est `string`.
Elle ne cherche pas à lire une chaîne finale complète : elle vérifie si la
génération peut continuer à l'intérieur d'une chaîne.

1. `lstrip(' \t\r\n')` retire seulement les espaces JSON **avant** le guillemet
   ouvrant. Les espaces à l'intérieur de la valeur restent présents.
2. Un texte vide est accepté : le prochain token peut encore ouvrir la chaîne.
3. Un texte non vide doit commencer par `"`.
4. Le cas simple, imprimable et sans guillemet interne ni antislash, est accepté
   directement. Cela évite une boucle Python pour les textes ordinaires.
5. Les cas plus complexes sont parcourus caractère par caractère.

`escaped` indique que le caractère précédent était un antislash : le caractère
suivant doit appartenir aux échappements JSON autorisés, comme `n`, `t`, `"`
ou `\`. Pour `\u`, `missing_digits` impose quatre chiffres hexadécimaux.

Une fin provisoire après un antislash ou après une partie de `\u00` reste
acceptée : le token suivant peut la compléter. Un guillemet non échappé dans
le corps signale une fermeture, donc ce n'est plus un préfixe **ouvert**.
La fonction renvoie alors `False` ; `string_completion` examinera si cette
fermeture constitue une fin acceptable.

```text
"Bonj       → début valide
"bonjour\  → début encore possible : échappement à compléter
"\u00      → début encore possible : deux chiffres hexadécimaux manquent
"bonjour\q → invalide : \q n'est pas un échappement JSON
"bonjour"  → chaîne fermée : à traiter par string_completion
```

Un retour à la ligne réel dans le corps est refusé ; les deux caractères `\n`
sont un échappement autorisé. Après lecture JSON, cet échappement peut devenir
un vrai retour à la ligne dans la valeur Python.

### Fonction `valid_prefix(text, kind) -> bool`

**Résumé rapide**

Elle choisit la règle de préfixe selon le type attendu de l'argument.
Elle permet de filtrer les tokens avant que le modèle choisisse parmi eux.

**Explication détaillée**

**Appelée par :** `decode_parameter`, pour chaque token candidat et chaque tour.
Le candidat examiné est le texte déjà généré **plus** le texte du token proposé.

Pour une chaîne, elle délègue à `string_prefix`.
Pour un nombre, elle vérifie d'abord les caractères possibles : chiffres,
signe moins et espaces JSON ; les décimaux et exposants ajoutent `.`, `+`,
`e`, `E` pour `number`. Une liste de caractères ne suffit cependant pas,
car `1.2.3` utiliserait uniquement des caractères permis.

Elle demande donc aussi :

```python
complete_number(text, kind) or complete_number(text + '0', kind)
```

Le premier test accepte un nombre déjà complet. Le second vérifie un début
auquel il manque un chiffre : `-` peut devenir `-0`, `1.` peut devenir `1.0`,
`1e-` peut devenir `1e-0`. Le zéro sert uniquement au **test** : il n'est ajouté
ni au contexte du modèle ni à la valeur finale.

Pour `integer`, un point ou un exposant est refusé dès le contrôle des
caractères. Pour `boolean`, seules les continuations menant à `true` ou `false`
sont acceptées, avec la gestion des espaces JSON autour du littéral.
`tru` est donc autorisé, `trux` et `True` ne le sont pas.

La fonction vérifie une possibilité syntaxique, pas la pertinence de la valeur
par rapport à la demande. Elle ne sait pas que l'utilisateur demandait `23`.

### Fonction `choose_token(current_ids, allowed, model) -> int`

**Résumé rapide**

Elle choisit le token ayant le meilleur score du LLM parmi les identifiants autorisés.
Si les règles ne laissent qu'un seul identifiant possible, elle le renvoie directement.

**Explication détaillée**

**Appelée par :** `select_valid_function` et `decode_parameter`.
`current_ids` est le contexte et `allowed` la liste des prochains tokens permis.
La sortie est **un identifiant de token**, pas son texte ni une valeur d'argument.

Le test `len(set(allowed)) == 1` gère aussi les doublons : plusieurs branches
peuvent proposer le même prochain identifiant. Lorsqu'un seul choix distinct
non négatif reste, appeler le modèle ne changerait rien au résultat.

Sinon, le SDK reçoit tout le contexte et renvoie les logits du prochain token.
`np.asarray` transforme cette liste de scores en tableau NumPy. Le code vérifie
ensuite que la liste autorisée n'est pas vide et que les identifiants sont dans
les limites du tableau ; une erreur déclenche une `ValueError`.

`scores = logits[allowed]` conserve seulement les scores des candidats permis.
Les scores non finis sont remplacés par `-np.inf`, afin de ne pas les sélectionner.
S'il ne reste aucun score fini, la fonction lève une erreur explicite.

Exemple avec des scores fictifs :

```text
allowed       = [7, 42, 99]
logits[7]     = 1.2
logits[42]    = 5.8
logits[99]    = 0.4
scores        = [1.2, 5.8, 0.4]
argmax(scores)= 1
allowed[1]    = 42  ← identifiant retourné
```

Il faut distinguer l'indice `1` dans la liste réduite et l'identifiant `42` du
vocabulaire. Le `return` effectue cette correspondance.
Une softmax n'est pas nécessaire : elle conserverait l'ordre des scores, alors
qu'on veut seulement le maximum. Le choix est glouton : le meilleur token
autorisé est retenu à ce tour, sans explorer plusieurs réponses complètes.

### Fonction `string_completion(text, suffix) -> tuple[str, str] | None`

**Résumé rapide**

Elle reconnaît une chaîne JSON terminée et une éventuelle portion de structure qui suit.
Elle conserve la partie déjà générée et indique quel suffixe reste à injecter.

**Explication détaillée**

**Appelée par :** `decode_parameter` lorsqu'un candidat n'est plus une chaîne
ouverte. `suffix` est la structure attendue après cette valeur, par exemple
`, "replacement":` ou `}`.

Après retrait des espaces JSON initiaux, `json.JSONDecoder().raw_decode(text)`
lit une première valeur et renvoie aussi la position de sa fin. À la différence
de `json.loads`, il permet d'examiner ce qui reste derrière cette valeur.
Le résultat doit être une chaîne. `value.encode('utf-8')` vérifie qu'elle est
encodable, notamment pour rejeter certains caractères Unicode isolés invalides.

`tail = text[end:]` contient la ponctuation déjà produite après le guillemet
fermant. Cette fin doit être un début du suffixe attendu. Le code accepte
le suffixe avec sa forme habituelle `, ` ou sans cet espace après la virgule.

Exemple :

```text
Texte candidat : "Alice",
Suffixe prévu : , "age":
Littéral lu   : "Alice"
Déjà présent  : ,
Reste à ajouter :  "age":
```

La fonction renvoie `(littéral_JSON, suffixe_restant)`, pas directement la
valeur Python. Elle renvoie `None` si la chaîne est incomplète, si le texte
supplémentaire n'est pas autorisé ou si l'encodage pose problème.

Pourquoi ce traitement ? Un seul token peut contenir la fin d'un mot,
le guillemet et la virgule. Refuser tous ces tokens enlèverait des sorties
naturelles du tokenizer ; ajouter le suffixe entier ensuite doublerait la
virgule. Cette fonction traite précisément cette frontière.

### Fonction `parse_scalar(text, kind) -> ParameterValue`

**Résumé rapide**

Elle transforme le littéral JSON terminé en une valeur Python du type attendu.
Elle conserve les entiers et les chaînes, et refuse un type final incompatible.

**Explication détaillée**

**Appelée par :** `decode_parameter`, une fois la génération de l'argument finie.
`json.loads(text)` effectue le décodage des échappements et des types JSON.

| `kind` | Exemple de texte JSON | Valeur retournée |
|---|---|---|
| `integer` | `23` | `23`, type `int` |
| `number` | `23` ou `23.5` | `23.0` ou `23.5`, type `float` |
| `string` | `"Alice"` | `Alice`, type `str` |
| `boolean` | `true` | `True`, type `bool` |

Pour `number`, la conversion `float(value)` est suivie d'un contrôle de finitude.
Pour les autres types, le code exige les types attendus puis renvoie la valeur.
Une incompatibilité provoque une `ValueError`, sans renvoyer arbitrairement `0`
ou une chaîne vide. Cette fonction est appelée après le filtrage de la grammaire
et avant `validate_call` : elle constitue une étape, pas une validation autonome
de tout l'appel de fonction.

Les espaces **dans** une chaîne sont conservés. Il n'y a pas de `.strip()` sur
la valeur finale des chaînes. Les flottants utilisent la précision habituelle
de Python : ce n'est pas un moteur de calcul décimal exact.

**Attention aux antislashs dans les regex :**

```json
"\b"
```

Ce littéral JSON représente un caractère de retour arrière. Pour obtenir les
deux caractères antislash et `b` dans la chaîne Python, le texte JSON doit être :

```json
"\\b"
```

Un moteur de regex peut ensuite interpréter ces deux caractères comme une
frontière de mot. `parse_scalar` respecte le JSON généré ; il ne peut pas deviner
qu'un retour arrière valide syntaxiquement était une erreur de raisonnement.

### Fonction `decode_parameter(current_ids, kind, pieces, model, suffix, max_tokens=256)`

**Résumé rapide**

Elle génère un argument token par token en ne proposant que des continuations autorisées.
Elle reconnaît sa fin, convertit sa valeur et complète le contexte avec la structure suivante.
Elle s'arrête avec une erreur si la limite de génération est atteinte.

**Explication détaillée**

**Appelée par :** `extract_all_parameters`, une fois par argument.
**Sortie :** une valeur scalaire Python.
**Effet supplémentaire :** `current_ids` est modifiée sur place, afin que la
génération suivante connaisse les arguments déjà produits.

**1. Initialiser.** `text = ''` pour tous les types, y compris les chaînes.
Le guillemet ouvrant d'une chaîne sera choisi avec son token, au lieu d'être
injecté à l'avance. Cela permet notamment un token contenant `"` et le début
d'un chemin dans le même morceau.

Le premier caractère de `suffix` est une virgule ou une accolade fermante.
Le SDK l'encode dans `stop`. Le code exige qu'il corresponde à un seul token,
car ce token sert de choix de terminaison pour les nombres.
`remaining` contient au départ tout le suffixe.

**2. Examiner les continuations.** À chaque tour, `allowed` et `endings` sont
recréés. Pour chaque token de `pieces`, le code calcule
`candidate = text + piece`.

- Si `valid_prefix(candidate, kind)` est vrai, le token reste autorisé.
- Sinon, pour une chaîne, `string_completion` peut accepter une fin de valeur
  et enregistrer le littéral ainsi que la structure restant à ajouter.
- Pour un nombre déjà complet, le token séparateur est ajouté à `allowed`.

La même règle de préfixe est donc testée très souvent. Cette boucle explique
une partie du coût de traitement, en plus des appels au modèle.

**3. Choisir.** `choose_token` sélectionne le meilleur token permis.
Pour un nombre, s'il choisit le séparateur, la valeur est finie et la boucle
s'arrête immédiatement. Le séparateur n'est pas ajouté à cet instant : il fait
partie du suffixe qui sera ajouté après la conversion.

**4. Prolonger ou terminer.** Dans les autres cas, le token est ajouté à
`current_ids`. S'il appartient à `endings`, la chaîne est terminée : le code
récupère son littéral et le suffixe restant, puis sort de la boucle.
Sinon, il ajoute le morceau à `text`. Un booléen égal à `true` ou `false`
est également terminé.

**5. Finaliser.** `parse_scalar` produit la valeur Python et le SDK encode le
suffixe restant pour compléter le contexte. La valeur est renvoyée à l'appelant.

Le `else` du `for` est exécuté si les 256 tours s'épuisent sans `break`.
La fonction lève alors une erreur : elle ne renvoie pas un fragment tronqué.
La limite porte sur le nombre de tours de génération d'**un argument**, pas
sur le nombre de demandes du fichier et pas sur 256 caractères.

**Pourquoi ne pas arrêter un nombre dès qu'il est valide ?** Après `1`,
le modèle peut encore vouloir produire `12`, `1.5` ou `1e3`. Le séparateur
doit entrer en concurrence avec les continuations numériques autorisées.
Une valeur déjà complète peut donc rester prolongeable.

### Fonction `extract_all_parameters(selected_func, current_ids, pieces, model)`

**Résumé rapide**

Elle parcourt les arguments de la fonction choisie, injecte leurs clés et génère leurs valeurs.
Elle renvoie le dictionnaire complet utilisé ensuite par `validate_call`.

**Explication détaillée**

**Appelée par :** `main`, après ajout de `, "parameters": {` au contexte.
`selected_func` donne les noms d'arguments et leurs types. Il ne contient pas
les valeurs à fournir ; celles-ci seront choisies pendant la génération.

1. `items = list(selected_func.parameters.items())` fixe la liste des
   paramètres, dans l'ordre conservé depuis leur définition JSON.
2. S'il n'y en a aucun, la fonction ajoute `}` au contexte et renvoie `{}`.
3. Sinon, elle encode la première clé et son deux-points, comme `"a":`.
4. Pour chaque argument, elle prépare le suffixe : soit la prochaine clé
   précédée d'une virgule, soit `}` pour le dernier argument.
5. Elle appelle `decode_parameter` avec le type décrit dans `detail.type`.
6. Elle stocke le résultat sous le nom de l'argument, puis passe au suivant.

`json.dumps` est utilisé pour les noms de clés afin de les échapper correctement.
Les clés et la ponctuation viennent du schéma ; les valeurs viennent des
choix du LLM dans l'ensemble autorisé. Le code n'attend donc pas que le modèle
invente la bonne orthographe des clés.

Pour `a` et `b`, le premier suffixe est `, "b":` ; le second est `}`.
Il n'y a pas de guillemet de valeur ajouté avec la clé : pour une chaîne,
ce guillemet fait partie de ce que `decode_parameter` génère.

Le contexte interne sert à prédire les arguments. Le fichier final ne résulte
pas d'une copie brute de ce contexte : il est produit depuis les objets Python
validés avec `json.dump`. En particulier, l'accolade finale des paramètres dans
ce contexte ne doit pas être confondue avec la sérialisation complète de l'objet
d'appel et du tableau final.

<a id="7-main"></a>

## 7. `src/__main__.py` : lancer et organiser le traitement

### Fonction `parse_args() -> argparse.Namespace`

**Résumé rapide**

Elle lit les options de la ligne de commande et fournit les trois chemins utilisés.
Sans option, elle utilise les fichiers d'entrée et de sortie prévus par défaut.

**Explication détaillée**

**Appelée par :** `main`, en premier dans son bloc `try`.
`argparse.ArgumentParser` définit la description du programme, puis les options :

| Option | Valeur par défaut |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calls.json` |

`parser.parse_args()` lit les arguments du processus et renvoie un
`argparse.Namespace`. On accède aux valeurs avec `args.input`, `args.output`
et `args.functions_definition`, pas comme à un dictionnaire.

Ces options sont des chaînes de caractères. Cette fonction ne vérifie pas
encore que les fichiers existent : la lecture les vérifiera ensuite.
Les chemins relatifs dépendent du dossier depuis lequel la commande est lancée.
Il faut donc se placer à la racine de `call_me` pour les valeurs par défaut.

`--help` affiche automatiquement l'aide et termine normalement avec le code 0.
Une option inconnue provoque le message d'argparse et le code 2. Ces sorties
utilisent `SystemExit`, qui n'est pas capturé par `except Exception` dans `main`.

### Fonction `write_results(path, results) -> None`

**Résumé rapide**

Elle crée le dossier de sortie si nécessaire et écrit la liste des appels en JSON UTF-8.
Elle produit un fichier lisible et refuse les nombres non finis lors de la sérialisation.

**Explication détaillée**

**Appelée par :** `main`, après un lot réussi ou immédiatement pour une liste
de demandes vide. **Entrées :** chemin de destination et liste de dictionnaires.
**Sortie Python :** `None`. Son résultat utile est le fichier écrit sur disque.

`output.parent.mkdir(parents=True, exist_ok=True)` crée les dossiers manquants.
`parents=True` permet plusieurs niveaux ; `exist_ok=True` accepte qu'ils existent.

Le fichier est ouvert en mode `'w'` : une sortie précédente à ce chemin est
remplacée. `json.dump` reçoit les options suivantes :

- `indent=2` : indentation de deux espaces, facile à lire.
- `ensure_ascii=False` : conservation lisible des caractères tels que `é`.
- `allow_nan=False` : refus de `NaN` et des infinis, incompatibles avec le JSON
  standard attendu.

Un retour à la ligne est ajouté à la fin du fichier. Ce n'est pas une écriture
atomique : le code ouvre directement le fichier final. Si l'écriture elle-même
échoue après ouverture, une sortie précédente peut avoir été tronquée.
En revanche, une erreur de génération avant cet appel n'écrit pas de nouveau
lot partiel, puisque les résultats sont d'abord accumulés en mémoire.

### Fonction `main() -> None`

**Résumé rapide**

Elle relie toutes les étapes, depuis les chemins d'entrée jusqu'au tableau JSON final.
Elle traite les demandes dans leur ordre avec un seul modèle chargé pour le lot.
Elle affiche les erreurs ordinaires sur stderr et termine alors avec le code 1.

**Explication détaillée**

**Appelée par :** le bloc final `if __name__ == '__main__': main()`.
Avec `python -m src`, Python exécute le module `src.__main__` comme point d'entrée.
La fonction ne renvoie pas la liste à l'utilisateur : elle organise la production
du fichier de sortie.

**Étape 1 — options et protection des fichiers.** Elle appelle `parse_args`,
puis compare les chemins résolus de sortie et des deux entrées.
`Path.resolve()` transforme notamment un chemin relatif en chemin absolu et
résout les liens symboliques. Si la sortie correspond à l'une des entrées,
le programme refuse de l'écraser. Il ne s'agit pas d'un contrôle général de
toutes les permissions du système de fichiers.

**Étape 2 — données d'entrée.** Elle appelle `validation_function`, construit
`functions_map`, puis appelle `load_prompts`.
`functions_map` associe directement chaque nom à son objet définition :
après la sélection, `functions_map[chosen_name]` retrouve la bonne définition.
Le refus des noms dupliqués rend cette association non ambiguë.

**Étape 3 — cas vide.** S'il n'y a aucune demande, elle écrit `[]` et revient.
Les imports ont déjà eu lieu, mais `Small_LLM_Model()` n'a pas été appelé :
les poids du modèle ne sont pas chargés pour ce cas.

**Étape 4 — initialisation commune.** Elle instancie le SDK, récupère le
vocabulaire avec `get_path_to_vocab_file`, le lit avec `json.load` et appelle
`build_vocabulary`. Ces opérations sont placées avant la boucle des demandes.

**Étape 5 — choix de fonction.** Pour chaque `item`, elle appelle
`select_valid_function(item.prompt, functions, model)`. La sélection dispose
de toutes les fonctions. Le nom choisi permet de récupérer `selected_func_def`.

**Étape 6 — contexte propre aux arguments.** Elle reconstruit ensuite un texte
contenant une consigne d'extraction, la seule définition choisie et la demande
originale. La consigne demande notamment de conserver ponctuation et guillemets
des chaînes. La réponse commence par le nom déjà fixé. Le SDK encode ce texte,
puis le code ajoute la transition `, "parameters": {`.

Ce nouveau `current_ids` remplace le contexte renvoyé par la sélection. Le modèle
continue donc avec moins de texte inutile pour générer les arguments, tout en
connaissant la demande, la fonction retenue et son schéma. Le nom n'est pas
redemandé au modèle à cette étape.

**Étape 7 — arguments et validation.** Elle appelle `extract_all_parameters`,
puis `validate_call`. L'objet validé est transformé par `model_dump` et ajouté à
`results`. Le `prompt` enregistré est toujours `item.prompt`.

**Étape 8 — sortie.** Lorsque toute la boucle a réussi, elle appelle
`write_results(args.output, results)`. La liste produite conserve l'ordre des
demandes du tableau d'entrée.

**Gestion des erreurs.** `except Exception as err` affiche `Error: ...` sur
`sys.stderr`, puis `sys.exit(1)` signale l'échec à l'environnement.
Une erreur interrompt le lot : le programme ne saute pas simplement la demande
qui pose problème. `KeyboardInterrupt`, déclenché par Ctrl+C, n'hérite pas
d'`Exception` mais de `BaseException` ; il n'est donc pas capturé par ce bloc.
Cela explique le traceback et le code 130 qu'on peut voir après une interruption.

### Pourquoi le programme semble silencieux

`main` n'affiche pas de compteur par demande ni de message final de succès.
Le SDK ou ses dépendances peuvent afficher le chargement du modèle et un
avertissement Hugging Face. Pendant les calculs, le terminal peut ensuite
rester sans nouvelle ligne.

Quand l'invite de commande revient sans erreur, inspecte le fichier de sortie.
La version actuelle ne met pas à jour ce fichier après chaque demande : elle
écrit à la fin. Pendant un nouveau lancement, un ancien fichier peut donc encore
être présent ; sa seule présence ne prouve pas que le nouveau lot est terminé.

<a id="8-sdk"></a>

## 8. `llm_sdk/llm_sdk/__init__.py` : le SDK fourni

Le SDK est une dépendance du projet, conservée dans sa version fournie.
Les explications ci-dessous servent à comprendre ses appels. L'application
utilise son interface publique et ne modifie pas ses attributs privés.

### Classe `Small_LLM_Model`

**Résumé rapide**

Elle regroupe le tokenizer et le modèle de langage local dans un même objet.
Ses méthodes permettent à l'application de passer du texte aux tokens et d'obtenir les scores suivants.

**Explication détaillée**

Cette classe fournie n'est pas un modèle Pydantic : elle représente un outil
d'inférence, pas un schéma de données JSON de l'application.
Ses attributs internes `_model`, `_tokenizer`, `_device`, `_dtype` et
`_model_name` sont utilisés par le SDK lui-même. Le préfixe `_` signale une
interface interne à ne pas manipuler depuis l'application.

Les bibliothèques principales sont `torch` pour les tenseurs et l'exécution
du réseau, `transformers` pour le chargement du tokenizer et du modèle, et
`huggingface_hub` pour trouver ou télécharger ses fichiers.
`logging.set_verbosity_error()` règle les messages du module de journalisation
de transformers ; cela ne supprime pas nécessairement tous les avertissements
des autres bibliothèques.

### Méthode `Small_LLM_Model.__init__(...) -> None`

**Résumé rapide**

Elle charge le tokenizer et le modèle Qwen, puis prépare le matériel et la précision de calcul.
Elle place le réseau en mode évaluation et désactive les gradients de ses paramètres.

**Explication détaillée**

**Appelée par :** `Small_LLM_Model()` dans `main`, une fois pour un lot non vide.
`__init__` est appelée lors de l'initialisation de l'instance ; elle ne renvoie
pas explicitement l'objet créé.

Le modèle par défaut est `Qwen/Qwen3-0.6B`. Le SDK accepte aussi `device`,
`dtype` et `trust_remote_code`. L'étoile `*` dans la signature impose de passer
ces options par leur nom. L'application ne les personnalise pas.

1. Si aucun matériel n'est demandé, il choisit MPS lorsqu'il est disponible,
   sinon CUDA, sinon le CPU.
2. Si aucune précision n'est demandée, il choisit `float16` pour CUDA/MPS,
   `float32` pour le CPU.
3. `AutoTokenizer.from_pretrained` charge le tokenizer. Si son identifiant de
   remplissage est absent, le SDK utilise l'identifiant de fin de séquence.
4. `AutoModelForCausalLM.from_pretrained` charge le réseau et ses poids.
5. `.to(device)` place le réseau sur le matériel sélectionné.
6. `.eval()` active le comportement d'évaluation, notamment pour les couches
   dont le fonctionnement dépend du mode entraînement/évaluation.
7. `requires_grad = False` désactive le calcul de gradients pour ses paramètres.

Le téléchargement initial peut coûter du temps et de l'espace. Les fichiers
du modèle déjà présents sont habituellement retrouvés dans le cache de
Hugging Face. Ce cache de fichiers est distinct d'un cache d'inférence.

### Méthode `Small_LLM_Model.encode(text) -> torch.Tensor`

**Résumé rapide**

Elle transforme un texte en identifiants de tokens avec le tokenizer fourni.
Elle renvoie un tenseur à deux dimensions placé sur le matériel du modèle.

**Explication détaillée**

**Appelée par :** la sélection et la génération pour les contextes, noms, clés
et suffixes. Elle utilise `add_special_tokens=False` : les marqueurs nécessaires
sont déjà présents dans le texte de conversation construit par l'application.

Le résultat a la forme `[ [id1, id2, ...] ]`, avec une dimension de lot de taille
1 et une dimension de séquence. Le type des identifiants est `torch.long`.

Pourquoi voit-on souvent `.tolist()[0]` ? `.tolist()` convertit le tenseur en
listes Python, et `[0]` récupère la seule séquence du lot. L'application obtient
ainsi `list[int]`, qu'elle peut prolonger avec `append` et `extend`.

Encoder un texte découpe ce texte en tokens ; cela ne demande pas au réseau
de raisonner ni de prédire la réponse. Le calcul du réseau a lieu dans la
méthode de logits.

### Méthode `Small_LLM_Model.decode(ids) -> str`

**Résumé rapide**

Elle transforme des identifiants de tokens en texte en utilisant le tokenizer.
Elle ignore les tokens spéciaux dans le texte retourné.

**Explication détaillée**

**Appelée par l'application :** `build_vocabulary`, pour décoder les tokens un
par un. Si l'entrée est un tenseur, la méthode le transforme d'abord en liste.
Elle appelle ensuite le décodage du tokenizer avec `skip_special_tokens=True`.

La méthode fournit une représentation lisible des morceaux du vocabulaire.
Un token isolé peut contenir seulement une partie des octets nécessaires à un
caractère ; cela explique le filtrage du caractère de remplacement dans
`build_vocabulary`. La méthode ne choisit pas de prochain token.

### Méthode `Small_LLM_Model.get_logits_from_input_ids(input_ids) -> list[float]`

**Résumé rapide**

Elle exécute le réseau sur le contexte et renvoie les scores bruts du prochain token.
Elle ne choisit aucun token ; `choose_token` applique ensuite les contraintes et le maximum.

**Explication détaillée**

**Appelée par :** `choose_token` lorsque plusieurs identifiants sont possibles.
La liste d'entrée est convertie en tenseur avec une dimension de lot de taille 1.
Le réseau est appelé dans un bloc `torch.no_grad()`, qui évite de construire
le graphe des gradients pendant cette inférence.

`out.logits[0, -1]` signifie : premier élément du lot, dernière position du
contexte, tous les scores du vocabulaire. Pour un modèle causal, ces scores
servent à prédire le token suivant. Ils sont convertis en liste de flottants.

L'identifiant d'un token sert d'indice dans cette liste. Le SDK renvoie les
scores bruts sans softmax. Le filtrage des tokens autorisés reste dans le code
de l'application, pas dans une version modifiée du SDK.

Le contexte complet courant est transmis à chaque appel de cette interface.
L'application n'ajoute pas de réutilisation d'état d'attention entre deux
appels. Il ne faut pas en déduire que toutes les couches de toutes les
bibliothèques sont dépourvues de mécanismes de cache internes.

### Méthode `Small_LLM_Model.get_path_to_vocab_file() -> str`

**Résumé rapide**

Elle récupère le chemin local du fichier de vocabulaire du modèle.
Elle peut le télécharger depuis Hugging Face s'il n'est pas déjà disponible.

**Explication détaillée**

**Appelée par :** `main`, avant `build_vocabulary`.
Le SDK cherche le nom `vocab_file` connu du tokenizer et utilise `vocab.json`
comme valeur de repli. `hf_hub_download` reçoit l'identifiant du modèle et ce nom.

La méthode retourne seulement un chemin. C'est `main` qui ouvre ensuite le
fichier et appelle `json.load`. Cette séparation évite de confondre récupération
d'un fichier et lecture de son contenu.

### Méthode `Small_LLM_Model.get_path_to_merges_file() -> str`

**Résumé rapide**

Elle fournit le chemin du fichier décrivant les fusions de tokens du tokenizer.
Elle fait partie du SDK, mais le programme actuel ne l'appelle pas.

**Explication détaillée**

Le fonctionnement ressemble à celui du vocabulaire : recherche du nom
`merges_file`, repli vers `merges.txt`, puis `hf_hub_download`.
Les fusions décrivent une partie de la construction des tokens pour les
tokenizers concernés. L'application laisse ce travail au tokenizer du SDK ;
elle ne lit pas ce fichier pour refaire l'algorithme de tokenisation.

### Méthode `Small_LLM_Model.get_path_to_tokenizer_file() -> str`

**Résumé rapide**

Elle fournit le chemin du fichier de description du tokenizer.
Elle est disponible dans le SDK mais n'intervient pas dans le parcours actuel de l'application.

**Explication détaillée**

Le SDK cherche le nom `tokenizer_file`, avec `tokenizer.json` comme repli,
puis obtient son chemin avec `hf_hub_download`.
Ce fichier décrit le tokenizer sérialisé. Puisque `encode` et `decode` assurent
déjà les conversions nécessaires, le programme ne l'ouvre pas directement.

**À retenir sur l'ordre :** les trois méthodes de chemins ne sont pas toutes
exécutées au démarrage. Seule `get_path_to_vocab_file` est appelée dans `main`.
La présence d'une méthode dans une classe ne signifie pas qu'elle est utilisée.

<a id="9-exemple"></a>

## 9. Une demande parcourue de bout en bout

Prenons un exemple pédagogique, sans prétendre qu'il reproduit les identifiants
réels du tokenizer ni une trace de mesures : `Add 2 and 3.`.

### Étape A — lecture et validation

Le fichier des définitions contient notamment :

```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": {"type": "number"},
    "b": {"type": "number"}
  },
  "returns": {"type": "number"}
}
```

Cet objet doit être un élément du tableau des définitions, pas nécessairement
le seul. Le fichier des demandes contient par exemple :

```json
[{"prompt": "Add 2 and 3."}]
```

`validation_function` construit l'objet définition. `load_prompts` construit
`PrompTest(prompt='Add 2 and 3.')`. À ce stade, rien n'a encore été additionné
ni extrait par le modèle.

### Étape B — sélection

Le LLM reçoit les descriptions disponibles et la demande. Le code limite les
tokens du nom aux branches autorisées. Supposons que la branche retenue soit
`fn_add_numbers`. Le programme récupère sa définition dans `functions_map`.

### Étape C — nouveau contexte et première clé

`main` reconstruit les consignes avec uniquement cette définition et la demande
originale. La fin du contexte contient maintenant :

```text
{"name": "fn_add_numbers", "parameters": {
```

`extract_all_parameters` ajoute la première clé :

```text
{"name": "fn_add_numbers", "parameters": {"a":
```

Ce texte est représenté par des identifiants dans `current_ids` ; le modèle ne
reçoit pas une variable Python `a` qui serait déjà renseignée.

### Étape D — génération de `a`

`decode_parameter` reçoit `kind='number'` et `suffix=', "b":'`.
Au départ `text=''`. Supposons qu'un token autorisé produise `2`.
Le texte est un nombre complet, mais peut encore être prolongé. La virgule de
fin est désormais autorisée à côté des continuations numériques possibles.
Si le modèle choisit de terminer, `parse_scalar('2', 'number')` renvoie `2.0`.
Le suffixe est ajouté au contexte, qui se termine alors par :

```text
... "parameters": {"a":2, "b":
```

La forme textuelle interne `2` et la valeur Python `2.0` peuvent donc différer
dans leur écriture tout en représentant le même argument numérique.

### Étape E — génération de `b`

Le même mécanisme génère `3` avec `suffix='}'`. La valeur Python obtenue est
`3.0`. Le dictionnaire renvoyé par `extract_all_parameters` est :

```python
{'a': 2.0, 'b': 3.0}
```

Si la définition avait exigé `integer`, les valeurs auraient été des `int`
et les décimaux ou exposants auraient été exclus par la règle correspondante.

### Étape F — validation et sortie

`validate_call` vérifie les clés `a`, `b`, les types numériques et la finitude.
Le résultat validé est transformé en dictionnaire, puis ajouté à `results`.
Après le traitement du lot, `write_results` produit :

```json
[
  {
    "prompt": "Add 2 and 3.",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

Il n'y a pas de `5` dans la sortie : on décrit un appel à effectuer.
Une bonne explication orale de cet exemple montre que tu as compris la
différence entre **générer des arguments** et **exécuter une fonction**.

<a id="10-outils"></a>

## 10. Environnement, Makefile et vérifications

### `pyproject.toml`, `uv.lock` et `.venv`

| Élément | Question à laquelle il répond |
|---|---|
| `pyproject.toml` | Quelles dépendances et quelles plages de versions le projet demande-t-il ? |
| `uv.lock` | Quelles versions exactes ont été résolues, y compris les dépendances indirectes ? |
| `.venv/` | Où sont installés le Python et les paquets de cet environnement local ? |

L'application déclare `llm-sdk`, NumPy et Pydantic. Le groupe de développement
déclare Flake8 et mypy. Les dépendances lourdes du SDK, telles que torch et
transformers, viennent de son propre `pyproject.toml` et sont résolues dans
l'environnement du projet racine.

`[tool.uv.sources]` indique que `llm-sdk` provient du dossier local `llm_sdk`,
avec `editable=true`. Le paquet importé pointe donc vers ces sources locales.
Cela permet de l'utiliser directement, mais ne signifie pas qu'il faut les
modifier. Le SDK fourni doit rester intact.

### Pourquoi `make run` utilise le venv sans l'activer dans le terminal

```bash
cd /home/evvan/Bureau/42School/Cercle_3/call_me
uv sync
uv run python -m src
```

`uv sync` crée ou synchronise `.venv` avec les dépendances du projet.
`uv run` choisit cet environnement pour la commande qu'il lance et assure la
synchronisation nécessaire selon son fonctionnement normal. L'activation
manuelle du shell n'est pas une condition pour utiliser `uv run`.

Cette commande, facultative, active l'environnement dans le terminal courant :

```bash
source .venv/bin/activate
```

Elle modifie notamment le `PATH` du shell. Elle n'est pas équivalente à
`uv sync`, qui installe/synchronise des paquets. Une recette Make s'exécute dans
un processus enfant : elle ne peut pas activer durablement l'environnement du
terminal parent. C'est pour cela que `make run` ne change pas ton invite en
sortant, alors qu'il a bien exécuté le Python du projet.

Pour vérifier le Python utilisé :

```bash
uv run python -c "import sys; print(sys.executable)"
```

### Toutes les cibles du Makefile

| Commande | Action réelle |
|---|---|
| `make install` | Exécute `uv sync`. |
| `make run` | Exécute `uv run python -m src`. |
| `make debug` | Exécute `uv run python -m pdb -m src` pour déboguer. |
| `make lint` | Lance `flake8 .`, puis mypy avec les options détaillées ci-dessous. |
| `make lint-strict` | Lance `flake8 .`, puis `mypy . --strict`. |
| `make clean` | Retire les caches Python/mypy/pytest trouvés hors des dossiers écartés, puis `data/output`. |

`.PHONY` signale que ces noms sont des actions, pas des fichiers à fabriquer.
Sans cible explicite, `make` choisit ici la première cible normale,
`install` : il synchronise l'environnement, sans activer ton terminal.
Il n'existe pas de cible `make test` dans cette version.

La cible `clean` préserve notamment `.venv`, `venv` et `.git` pendant sa recherche.
Elle supprime les sorties : copie un résultat que tu veux conserver avant de
l'utiliser.

### Flake8 et mypy ne contrôlent pas la même chose

Flake8 détecte des problèmes de style et certaines erreurs statiques : imports
inutilisés, indentation, lignes trop longues, etc. `.flake8` fixe une longueur
maximale de 79 caractères et exclut le SDK et les dossiers générés.

mypy vérifie la cohérence des annotations Python. Dans `make lint`, les options
sont :

| Option | Rôle |
|---|---|
| `--warn-return-any` | Signaler certains retours `Any` depuis une fonction au type de retour plus précis. |
| `--warn-unused-ignores` | Signaler les commentaires `type: ignore` devenus inutiles. |
| `--ignore-missing-imports` | Ne pas bloquer sur les bibliothèques sans informations d'import/types utilisables. |
| `--disallow-untyped-defs` | Exiger des annotations pour les définitions de fonctions. |
| `--check-untyped-defs` | Vérifier aussi le corps des fonctions non annotées, s'il y en a. |

`--strict` active un ensemble de contrôles plus exigeants dans la seconde cible.
Cela ne vérifie pas si Qwen a correctement compris la demande.

`mypy_path` pointe vers le sous-projet local pour résoudre l'import `llm_sdk`.
Les exclusions actuelles évitent de découvrir le SDK et `.venv` comme fichiers à
analyser. Le réglage `follow_imports='skip'` pour `llm_sdk` empêche aussi mypy
d'en analyser le contenu en suivant cet import. Les appels du programme sont
vérifiés dans les limites des types disponibles ; le SDK n'est pas audité par
ces cibles. Exclure une dépendance du contrôle ne change pas son exécution.
Dans les réglages présents à la fin de cette rédaction, `moulinette/` ne figure
plus dans les exclusions : ses fichiers Python encore présents peuvent donc
être analysés par les commandes portant sur `.`. Ce dossier externe n'est pas
une dépendance nécessaire à l'application.

Les derniers contrôles de code réalisés le 17 novembre, avant ce guide, ont
réussi pour `make lint` et `make lint-strict`, avec cinq fichiers Python de
l'application, lorsque la moulinette était exclue. Les exclusions ont changé
pendant la rédaction du guide ; ces résultats antérieurs ne sont donc pas une
nouvelle validation du périmètre élargi actuel. Cette rédaction documentaire
n'est pas non plus une nouvelle exécution des lots de génération ; les résultats
mesurés sont distingués en partie 11.

### Commandes utiles pour utiliser le projet

Depuis la racine du projet :

```bash
make install
make lint
make lint-strict
make run
python3 -m json.tool data/output/function_calls.json
```

`json.tool` vérifie et affiche le JSON, mais ne vérifie pas le sens des arguments.
Pour choisir d'autres chemins, lance directement la commande Python :

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/custom_output.json
```

`make run --input ...` ne transmet pas automatiquement une option à Python :
le Makefile ne prévoit pas ce mécanisme. Utilise la commande ci-dessus.

### Messages fréquents et leur cause

| Situation | Explication |
|---|---|
| Avertissement de requêtes Hugging Face non authentifiées | Pas de token d'authentification ; le modèle public peut rester accessible, avec des limites de téléchargement différentes. Ce n'est pas une preuve d'erreur dans les arguments. |
| `Loading weights` | Chargement des poids du modèle, distinct de la génération des appels. |
| Terminal sans progression | L'application n'imprime pas de compteur pendant la génération. |
| `Cannot read ...` | Fichier absent, permission, encodage ou JSON incorrect ; lire la cause affichée. |
| `expected a JSON array` | Le document commence par un objet ou une autre valeur au lieu d'un tableau. |
| Erreur Pydantic sur un champ | Définition ou demande non conforme au modèle attendu. |
| `Function names must be unique` | Deux définitions ont le même nom. |
| `Output must not overwrite either input file` | Le chemin de sortie résolu correspond à une entrée. |
| `generation exceeded 256 tokens` | L'argument n'a pas atteint une fin autorisée dans la limite. |
| `No valid token ...` | Aucun choix de token compatible avec les contrôles de grammaire. |
| `KeyboardInterrupt` après Ctrl+C | Arrêt manuel pendant les calculs, pas diagnostic d'une mauvaise compréhension de la demande. |

<a id="11-limites"></a>

## 11. Changements à comprendre et limites de la version actuelle

### Les corrections fonctionnelles importantes

| Sujet | Problème traité | Fonctionnement actuel |
|---|---|---|
| Entiers | Une définition `integer` n'était pas prise en charge ; un flottant ne remplace pas toujours un entier attendu. | Alias, génération, conversion et validation distinguent `integer` et `number`. |
| Ouverture des chaînes | Un guillemet injecté seul pouvait écarter un token naturel contenant le guillemet et le début de la valeur. | `decode_parameter` commence avec un texte vide et laisse générer l'ouverture ; la clé est injectée sans guillemet de valeur. |
| Espaces avant les chaînes | La grammaire devait autoriser les espaces JSON autour de la syntaxe. | Les espaces avant le guillemet sont acceptés sans supprimer les espaces du contenu. |
| Vitesse des préfixes simples | La boucle caractère par caractère était répétée pour de nombreux candidats ordinaires. | Le cas imprimable sans échappement est accepté rapidement ; les cas complexes gardent leur contrôle détaillé. |
| Longueur du contexte d'arguments | Les définitions non choisies alourdissaient les appels au modèle pendant les arguments. | Après sélection parmi toutes les fonctions, le contexte est reconstruit pour la seule définition retenue. |

Le choix du nom reste fondé sur les logits du LLM et les valeurs restent générées
par le modèle. Aucun numéro de test ni réponse attendue n'est lu pour fabriquer
les arguments. Le SDK fourni et `function_select.py` sont restés inchangés lors
du dernier correctif. `parser.py` conserve l'ajout des entiers précédemment autorisé.

Il n'y a pas d'ajout de cache d'inférence, de traitement par lots parallèle ou de
réimplémentation du tokenizer. La table des morceaux du vocabulaire et le cache
de téléchargement Hugging Face ont des rôles différents, expliqués plus haut.

### Résultats effectivement mesurés le 17 novembre 2026

| Lot disponible | Appels jugés corrects | Conformité structurelle | Durée totale |
|---|---|---|---|
| Public | **10/11 — 90,9 %** | **11/11** | **162,62 s, environ 2 min 43 s** |
| Privé utilisé pour le diagnostic | **10/11 — 90,9 %** | **11/11** | **173,96 s, environ 2 min 54 s** |

Les lots complets ont été générés avec le SDK d'origine puis évalués avec
l'outil fourni. Les durées incluent le chargement du modèle avec ses poids
déjà téléchargés. Le matériel, les versions installées et un premier
téléchargement peuvent changer la durée sur une autre machine.
Ces scores sont des mesures sur ces lots, pas une garantie de 100 % sur une
autre évaluation ni une promesse de note finale.

### Les deux erreurs sémantiques encore connues

**Remplacement de nombres par regex :** dans un cas public, la regex contient
des caractères de retour arrière là où des frontières de mot étaient attendues.
La chaîne est du JSON valide et de type `string`, mais le remplacement ne
produit pas le texte souhaité. La distinction entre `"\b"` et `"\\b"` expliquée
dans `parse_scalar` permet de comprendre ce problème.

**Copie de guillemets internes :** dans un cas de diagnostic, un texte contenant
`Say "hello" to {name}` a été recopié sans les guillemets autour de `hello`.
La chaîne reste valide mais son contenu n'est pas identique à la demande.
La consigne de copie exacte réduit les erreurs sans les rendre impossibles.

### Ce que le programme garantit par ses contrôles, et ce qu'il ne garantit pas

- La sélection limite les noms aux fonctions fournies. Elle ne prouve pas
  que la fonction choisie est la plus pertinente pour une demande ambiguë.
- Les clés des arguments viennent du schéma et sont revérifiées. Les types
  sont contrôlés à plusieurs étapes ; cela ne prouve pas l'exactitude des valeurs.
- Les nombres sont contrôlés pour rester finis ; aucune contrainte métier comme
  « âge positif » ou « racine carrée d'un nombre positif » n'est déclarée ici.
- Les types pris en charge sont les quatre scalaires annoncés. Les tableaux,
  objets imbriqués, `null`, paramètres optionnels et valeurs par défaut ne font
  pas partie du schéma implémenté.
- Tous les paramètres déclarés sont requis. Aucune recherche dans une liste
  fixe de onze demandes n'est utilisée : la boucle traite la liste fournie.
- La génération est bornée par argument, mais une erreur peut interrompre le lot.
  Il n'y a ni reprise automatique ni nouvel essai de réponse dans ce code.
- Le fichier final est sérialisé depuis les valeurs Python, ce qui évite de
  recopier une conversation brute. L'écriture finale reste directe, non atomique.

<a id="12-revision"></a>

## 12. Réviser avant la soutenance

### Une présentation orale d'environ une minute

> Mon programme reçoit des descriptions de fonctions et des demandes en JSON.
> Je valide d'abord ces entrées avec Pydantic, avant de charger le modèle.
> Pour chaque demande, le LLM choisit un nom parmi les fonctions disponibles,
> avec un filtrage des tokens qui empêche d'inventer un autre nom.
> Je reconstruis ensuite un contexte limité à la fonction choisie.
> Le code impose les clés des arguments, et le LLM génère leurs valeurs token
> par token sous les contraintes du type attendu.
> Je valide les clés, les types et les nombres finis, puis j'écris tous les
> appels dans un tableau JSON, dans l'ordre d'entrée.
> Le programme prépare les appels ; il n'exécute pas les fonctions métier.
> Le SDK fourni s'occupe du modèle et du tokenizer et reste inchangé.

### Questions que tu dois pouvoir expliquer sans lire tout le code

**1. Quel fichier démarre avec `python -m src` ?**
`src/__main__.py`. Le bloc `if __name__ == '__main__'` appelle `main`.

**2. Quel fichier valide les données ?**
`parser.py`, avec ses modèles Pydantic et `validate_call`.

**3. Où est fait le choix sémantique d'une fonction ?**
Dans les scores du modèle consultés pendant `select_valid_function`.
Le Python ne détecte pas des mots-clés pour imposer une fonction métier.

**4. Pourquoi ne pas laisser le modèle écrire tout le JSON librement ?**
Il pourrait inventer un nom, oublier une clé ou casser la syntaxe.
Le code impose la structure connue et filtre les continuations des valeurs.

**5. À quoi sert `allowed` ?**
À lister les identifiants autorisés pour le prochain token. Le maximum est
choisi seulement parmi eux, même si un token interdit a un score global supérieur.

**6. Pourquoi une softmax n'est-elle pas nécessaire ?**
On veut le score maximal ; la softmax ne changerait pas l'ordre des scores.

**7. Quelle différence entre `current_ids` et `text` ?**
`current_ids` contient les tokens de tout le contexte courant. `text` contient
seulement le littéral JSON de l'argument en cours.

**8. Pourquoi `current_ids` change-t-elle dans une fonction qui ne la renvoie pas ?**
C'est une liste mutable passée à la fonction. `append` et `extend` modifient
le même objet que l'appelant utilise. Une réaffectation locale n'aurait pas le
même effet ; ici la génération enrichit effectivement la liste partagée.

**9. Pourquoi une chaîne ouverte est-elle autorisée alors qu'elle n'est pas encore du JSON complet ?**
Parce que la génération se fait progressivement. Il faut autoriser les débuts
qui pourront devenir valides, puis reconnaître séparément une fin correcte.

**10. Pourquoi tester `text + '0'` ?**
Pour vérifier qu'un préfixe numérique incomplet peut être terminé par un chiffre.
Ce zéro n'est pas une valeur inventée pour remplacer la réponse du modèle.

**11. À quoi sert `string_completion` en plus de `string_prefix` ?**
La première accepte une chaîne finie et la structure déjà présente derrière.
La seconde vérifie uniquement une chaîne encore ouverte. Un token peut traverser
la frontière entre contenu et ponctuation JSON.

**12. Pourquoi `integer` ne se contente pas de `number` ?**
Le destinataire peut exiger un vrai entier Python, par exemple pour certaines
opérations de formatage ou d'indexation. `23.0` est un `float`, pas un `int`.

**13. Pourquoi `True` doit-il être exclu d'un `number` ?**
Parce qu'en Python `bool` hérite de `int`, alors que le schéma distingue ces types.

**14. `ConfigDict` remplace-t-il toutes les conditions de `validate_call` ?**
Non. Il configure chaque modèle, mais la validation des clés et types propres
à la fonction choisie nécessite aussi la comparaison avec sa définition.

**15. Combien de fois le modèle est-il chargé ?**
Une fois par lancement contenant au moins une demande, avant la boucle.
Il y a ensuite plusieurs inférences par demande, généralement une par choix
de token qui nécessite de comparer des scores.

**16. Les onze demandes sont-elles codées en dur ?**
Non. `main` parcourt `prompts` et les paramètres de la définition sélectionnée.
Changer le tableau d'entrée ne nécessite pas d'ajouter une branche par test,
dans les limites des types pris en charge.

**17. Pourquoi le contexte est-il reconstruit après la sélection ?**
Toutes les définitions sont utiles pour choisir. Une fois le nom fixé,
la génération des arguments n'a besoin que de la définition retenue et de la
demande. Réduire ce texte diminue le travail répété du modèle.

**18. Le fichier `fn_add_nb.py` additionne-t-il les nombres ?**
Non. Il génère les valeurs des arguments de toutes les fonctions prises en charge.
L'addition réelle n'appartient pas au programme de préparation des appels.

**19. Un JSON conforme signifie-t-il que la réponse est juste ?**
Non. Une regex peut être une chaîne valide tout en exprimant le mauvais motif.
C'est la différence entre validation structurelle et justesse sémantique.

**20. Pourquoi `make run` ne m'affiche-t-il pas l'activation du venv ?**
Parce que la commande utilise `uv run`, qui choisit l'environnement pour le
processus Python. Elle ne change pas l'état du shell parent.

**21. Que font Flake8 et mypy par rapport au modèle ?**
Ils vérifient le code et ses types, pas la compréhension des demandes par Qwen.
Le SDK fourni est exclu de leurs analyses dans la configuration actuelle.

**22. Que se passe-t-il en cas d'erreur sur la dernière demande ?**
Le lot s'arrête avec un message et un code de sortie d'échec. Comme la sauvegarde
est après la boucle, les résultats de ce lancement ne sont pas écrits en lot
partiel. Un ancien fichier peut toujours être présent.

### Plan de révision pratique d'environ 45 minutes

1. **5 minutes :** raconter le but et la différence appel/résultat, puis regarder
   l'arborescence et l'ordre d'exécution.
2. **10 minutes :** suivre `main` et l'exemple d'addition, en montrant quel fichier
   contient chaque étape.
3. **10 minutes :** expliquer `select_valid_function` et `choose_token`, avec
   l'exemple de branches et l'exemple de scores.
4. **10 minutes :** expliquer `decode_parameter`, les chaînes et les suffixes ;
   comprendre la différence entre un préfixe possible et une valeur terminée.
5. **5 minutes :** revoir Pydantic, les quatre types et `validate_call`.
6. **5 minutes :** expliquer uv, les deux cibles lint et les limites mesurées,
   puis répondre à voix haute aux questions les moins claires.

Pour vérifier ta compréhension, change seulement l'exemple sur papier : choisis
une demande avec une chaîne, puis une définition avec un `integer`. Explique
quelles étapes restent identiques et quelles règles de type changent.
Si tu sais suivre ces deux cas sans confondre structure, tokens et sens des
valeurs, tu disposes des repères essentiels pour présenter ce code.
