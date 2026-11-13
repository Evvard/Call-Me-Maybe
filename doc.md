# Call Me Maybe — version obligatoire simplifiée

Guide en français pour **eolivier**, mis à jour le **13 novembre 2026**.

Cette version vise la partie obligatoire du sujet 1.7. Les fichiers
`src/parser.py` et `src/function_select.py` ont été conservés exactement tels
qu’ils étaient au début de la simplification : leur contenu est vérifié par
empreinte SHA-256. Les commentaires et docstrings restent en anglais.

Le cache d’inférence ajouté précédemment a été supprimé. Il n’y a pas de
batching, de tokenizer réécrit, de génération graphique, d’arguments imbriqués,
de mécanisme de récupération avancée ni de suite de tests complète ajoutée au
rendu. Les contrôles nécessaires à la partie obligatoire sont conservés.

## 1. Le travail demandé

À partir de :

```text
What is the sum of 2 and 3?
```

le programme doit produire un appel :

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

Il ne calcule pas la somme et n’exécute pas la fonction. Le LLM sélectionne
l’outil et extrait les arguments ; le programme impose la structure attendue.

Il faut distinguer trois contrôles :

- **Syntaxe** : le fichier est-il un JSON lisible ?
- **Schéma** : possède-t-il exactement les bonnes clés et les bons types ?
- **Sens** : le nom de fonction et les valeurs répondent-ils à la demande ?

Un appel contenant `a=7.0` et `b=9.0` reste bien typé, mais serait incorrect
pour une question portant sur 2 et 3. Pydantic ne vérifie pas la compréhension
du texte par le modèle.

## 2. Les fichiers

| Fichier | Rôle |
|---|---|
| `src/__init__.py` | Marque le paquet de l’application ; il peut rester vide. |
| `src/__main__.py` | Arguments de commande, orchestration et écriture du résultat. |
| `src/parser.py` | Validation Pydantic des entrées et des appels ; non modifié. |
| `src/function_select.py` | Choix d’une fonction par les logits ; non modifié. |
| `src/fn_add_nb.py` | Génération contrainte des arguments scalaires. |
| `data/input/functions_definition.json` | Fonctions proposées, paramètres et types. |
| `data/input/function_calling_tests.json` | Questions à traiter, dans leur ordre. |
| `llm_sdk/llm_sdk/__init__.py` | SDK fourni : chargement de Qwen, encodage, décodage et logits. |
| `llm_sdk/pyproject.toml` | Dépendances du SDK local. |
| `llm_sdk/uv.lock` | Verrou du sous-projet SDK ; le projet utilise le verrou racine. |
| `pyproject.toml` | Dépendances et configuration mypy de l’application. |
| `uv.lock` | Versions résolues pour installer l’ensemble du projet. |
| `makefile` | Installation, lancement, débogage, lint et nettoyage. |
| `.flake8` | Exclusion du SDK fourni, des environnements et caches ; lignes de 79 caractères. |
| `.gitignore` | Exclusion des environnements, caches et résultats de Git. |
| `test/test.py`, `test/tests.py`, `test/testz.py` | Expériences manuelles historiques. |
| `README.md` | Présentation en anglais exigée par le sujet. |
| `doc.md` | Explication française du code actuel. |
| `en.subject.pdf` | Sujet local de référence, version 1.7. |

Le résultat par défaut est `data/output/function_calls.json`. C’est un fichier
généré, exclu de Git. `.venv`, `.mypy_cache` et `__pycache__` ne sont pas des
sources à remettre.

## 3. Le chemin d’une demande

```text
Fichiers JSON
    → validation Pydantic
    → chargement de Qwen
    → lecture du vocabulaire et décodage par le SDK
    → sélection contrainte du nom
    → génération contrainte des arguments
    → vérification des clés et des types
    → écriture du tableau JSON
```

Le modèle est chargé une seule fois pour le lot. Les demandes sont traitées
l’une après l’autre ; il n’y a pas de traitement par lots dans le modèle.

## 4. `src/parser.py` — modèles et validation

### 4.1 Les alias `ParameterType` et `ParameterValue`

`ParameterType` n’est pas une fonction : c’est la liste des types pris en charge dans le schéma, soit `number`, `string` et `boolean`.

`ParameterValue` décrit les valeurs manipulées dans le programme : chaînes Python, nombres flottants et booléens.

Cette distinction est utile : le texte `"number"` décrit un type dans la définition ; la valeur `2.0` est un argument réel de ce type.

### 4.2 Classe `InputDetail`

Elle décrit un argument obligatoire :

```json
{"type": "number"}
```

Elle hérite de `BaseModel`, donc Pydantic vérifie les champs lors de sa création.

- `extra='forbid'` interdit les champs inattendus.
- `strict=True` empêche les conversions implicites de types.
- Le champ `type` utilise `ParameterType`, donc `"numbre"` est refusé immédiatement.

Avant, `type: str` acceptait n’importe quel texte. Le programme pouvait ensuite traiter un type mal orthographié comme une chaîne à cause d’un `else` trop large.

### 4.3 Classe `OutputDetail`

Elle hérite de `InputDetail` et valide le type de retour déclaré par une fonction. Elle ne représente pas le résultat d’une exécution : le projet n’exécute pas les fonctions.

L’héritage évite de dupliquer les mêmes règles de validation des types scalaires.

### 4.4 Classe `FonctionDefinition`

Elle représente une fonction complète, avec :

- `name` : un nom non vide ;
- `description` : son rôle, présenté au modèle ;
- `parameters` : dictionnaire des arguments et de leurs types ;
- `returns` : type de retour annoncé.

Le dictionnaire vide `{}` est permis pour une fonction sans argument. En revanche, le champ `parameters` doit bien être présent dans sa définition.

Le nom français de la classe a été conservé pour éviter un renommage inutile de l’interface existante.

### 4.5 Classe `PrompTest`

Elle contient exactement un champ `prompt`, de type chaîne. Le texte original est conservé pour être recopié dans la sortie.

Une chaîne vide est acceptée comme cas limite. Un nombre donné à la place du texte est rejeté, sans conversion automatique en chaîne.

### 4.6 Classe `FunctionCallResult`

Elle représente les trois champs de sortie exigés : `prompt`, `name` et `parameters`.

**Attention : cette classe seule ne connaît pas la définition de la fonction choisie.** C’est pourquoi le programme doit passer par `validate_call()` avant de créer le résultat final. Une union de types scalaires ne suffit pas à vérifier que l’argument `a` est précisément un nombre pour telle fonction.

### 4.7 Fonction `load_json_list(path_json)`

**Entrée :** chemin d’un fichier, sous forme de chaîne ou de `Path`.

**Sortie :** tableau Python dont les éléments ne sont pas encore validés par les modèles métier.

Étapes :

1. transformer le chemin en objet `Path` ;
2. ouvrir le fichier en UTF-8 avec `with`, pour garantir sa fermeture ;
3. lire son JSON ;
4. transformer les erreurs de lecture, de décodage ou de syntaxe en message contenant le chemin ;
5. vérifier que la racine du JSON est un tableau.

Un fichier contenant `{}` est un JSON syntaxiquement valide, mais ce n’est pas le format d’entrée attendu. Il doit donc être refusé.

Le `Any` de cette fonction est volontaire : juste après `json.load()`, les éléments peuvent encore avoir n’importe quelle forme. Pydantic effectue ensuite la validation précise.

### 4.8 Fonction `validation_function(path_json)`

**Entrée :** chemin des définitions de fonctions.

**Sortie :** liste de `FonctionDefinition` validées.

Elle utilise `load_json_list()`, puis valide chaque élément avec `FonctionDefinition.model_validate()`.

Deux contrôles supplémentaires sont nécessaires :

- une liste vide ne permettrait aucun choix de fonction ;
- deux fonctions portant le même nom rendraient le dispatch ambigu et s’écraseraient dans un dictionnaire indexé par nom.

La fonction rejette ces deux cas avant le chargement du modèle, ce qui évite de charger des poids lourds pour une entrée déjà inutilisable.

### 4.9 Fonction `load_prompts(path_json)`

**Entrée :** chemin du tableau des demandes.

**Sortie :** liste de `PrompTest` dans l’ordre initial.

La lecture du fichier est partagée avec `load_json_list()`. Chaque élément est ensuite validé avec Pydantic.

La liste vide est permise : elle signifie qu’il n’y a aucune question à traiter. Le programme écrira alors `[]` sans charger le modèle.

### 4.10 Fonction `validate_call(prompt, definition, parameters)`

**Entrées :** demande originale, fonction sélectionnée et valeurs extraites.

**Sortie :** un `FunctionCallResult` prêt à être sérialisé.

Cette fonction constitue le contrôle final du schéma :

1. comparer exactement les clés obtenues aux clés de `definition.parameters` ;
2. vérifier chaque valeur selon le type déclaré ;
3. refuser les nombres infinis et `NaN` ;
4. construire seulement ensuite le modèle de sortie.

Exemples :

| Valeur pour un argument `number` | Résultat |
|---|---|
| `2.0` | Acceptée. |
| `"2"` | Refusée : c’est du texte. |
| `True` | Refusée : c’est un booléen. |
| `float('inf')` | Refusée : non fini. |
| `float('nan')` | Refusée : non fini. |

Pourquoi exclure explicitement les booléens ? En Python, `bool` hérite de `int`. Un simple `isinstance(True, int)` vaut donc `True`, alors que le schéma JSON distingue les deux types.

Le paramètre est annoté `dict[str, ParameterValue]` dans la version actuelle de `parser.py`. Le dictionnaire reçu est lu pour vérifier les clés et les valeurs ; `dict(parameters)` crée ensuite une copie pour le modèle final.

## 5. `src/function_select.py` — choisir la fonction

### 5.1 Fonction `select_valid_function(prompt, functions, model)`

**Entrées :** demande utilisateur, définitions disponibles et modèle SDK.

**Sortie :** couple `(nom_choisi, contexte_en_tokens)`.

Elle ne cherche pas des mots-clés dans la demande pour décider du nom. Le choix dépend des scores du LLM.

#### Étape A — construire le contexte

Le message système présente les définitions et demande de produire un appel JSON. Il précise de copier les arguments de la question et de ne pas exécuter les fonctions.

Le contexte contient ensuite la demande utilisateur, puis le début de réponse :

```text
{"name":
```

Les marqueurs de conversation correspondent au format de Qwen. Le SDK transforme ce texte en identifiants de tokens.

#### Étape B — préparer les branches autorisées

Chaque nom est d’abord transformé en chaîne JSON complète par `json.dumps()`, puis tokenisé.

Exemple conceptuel :

```text
"fn"
"fn_long"
```

Les guillemets font partie des branches. C’est la correction qui permet au nom court de se terminer sans être forcé à continuer vers le nom long.

Le nombre réel de tokens n’est pas le nombre de caractères. Le tokenizer peut regrouper plusieurs caractères dans un seul token.

#### Étape C — filtrer les branches à chaque tour

`generated` contient le préfixe déjà choisi. Le programme conserve uniquement les noms dont la séquence de tokens commence par ce préfixe.

- Si une séquence complète correspond au préfixe, la fonction est sélectionnée.
- Sinon, les tokens suivants possibles de toutes les branches restantes sont autorisés.
- `choose_token()` fait choisir au modèle le meilleur token parmi eux.
- Ce token est ajouté à la fois à `generated` et au contexte.

Le contexte retourné inclut le guillemet final du nom. Le programme principal ajoute donc `, "parameters": {` sans rajouter un deuxième guillemet.

## 6. `src/fn_add_nb.py` — génération des arguments

Le nom du fichier est historique : il traite tous les arguments scalaires.
Il n’y a plus d’import `re`, ni de règle écrite en expression régulière dans
ce fichier. Les nombres sont lus avec `json.loads()` ; les chaînes incomplètes
sont parcourues avec une boucle. Les docstrings du code sont courtes ; les
explications détaillées sont ici.

### 6.1 `build_vocabulary(vocab, model)`

**Entrées :** le vocabulaire `texte → identifiant` et le SDK.
**Sortie :** un dictionnaire `identifiant → texte décodé`.

La boucle appelle `model.decode([token_id])` pour chaque token ordinaire.
Le SDK effectue donc le décodage. Les marqueurs spéciaux, textes vides et textes
contenant le caractère de remplacement `�` sont écartés.

Un token représente parfois seulement une partie d’un caractère Unicode. Il
peut alors être exclu par cette méthode simple. Les échappements JSON comme
`\u00e9` restent possibles. Ce dictionnaire ne conserve aucun résultat du LLM
et aucun état d’attention : ce n’est pas un cache d’inférence.

### 6.2 `complete_number(text)`

**Entrée :** un texte. **Sortie :** `True` si c’est un nombre JSON fini.

La fonction essaie de lire le texte avec `json.loads()`. Elle vérifie ensuite
que le résultat est exactement un `int` ou un `float`, puis que
`math.isfinite()` accepte sa valeur. Un JSON invalide ou un nombre trop grand
renvoie `False`.

Exemples : `-12.5` est accepté ; `1.` est incomplet ; `01` et `--2` sont
invalides. `true` est un booléen, donc pas un nombre. `NaN` et `1e309` sont
refusés, car le résultat ne serait pas fini.

### 6.3 `string_prefix(text)`

**Entrée :** le texte d’une chaîne en cours de génération, avec son guillemet
initial. **Sortie :** `True` si elle peut encore être complétée.

La boucle parcourt les caractères avec deux variables :

- `escaped` : le caractère précédent était un antislash ;
- `missing_digits` : nombre de chiffres hexadécimaux encore attendus après `\u`.

Dans un échappement Unicode, seuls `0` à `9`, `a` à `f` et `A` à `F` sont
acceptés. Après un antislash, seuls les échappements JSON sont acceptés :
`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t` et `\uXXXX`.
Dans le texte normal, un contrôle brut ou un guillemet non échappé termine
ce préfixe ouvert et renvoie `False`.

Un antislash final ou un `\u12` incomplet reste autorisé : les tokens suivants
peuvent le compléter. La fermeture de la chaîne est reconnue séparément par
`string_completion()`. Par exemple, `"bonjour` est un préfixe ouvert ;
`"bonjour"` est une chaîne complète.

### 6.4 `valid_prefix(text, kind)`

**Entrées :** texte candidat et type attendu. **Sortie :** peut-on continuer ?

Pour une chaîne, elle appelle `string_prefix()`. Pour un booléen, elle vérifie
que `true` ou `false` commence par le texte : `tru` passe, `trux` échoue.
Les espaces JSON autour d’un booléen sont acceptés.

Pour un nombre, elle écarte d’abord les caractères étrangers à un nombre JSON.
Cette vérification seule serait insuffisante : `1.2.3` contient uniquement des
caractères numériques, mais reste invalide. Elle demande donc aussi :

```python
complete_number(text) or complete_number(text + '0')
```

Pourquoi ajouter `0` ? Un nombre inachevé courant peut être complété par un
chiffre. Cette vérification permet de conserver les tokens de début :

| Texte | Texte suivi de `0` | Résultat |
|---|---|---|
| `-` | `-0` | Préfixe accepté. |
| `1.` | `1.0` | Préfixe accepté. |
| `1e-` | `1e-0` | Préfixe accepté. |
| `01` | `010` | Toujours invalide, rejeté. |
| `1.2.3` | `1.2.30` | Toujours invalide, rejeté. |

Le zéro sert uniquement au contrôle. Il n’est ajouté ni au contexte du modèle,
ni à la valeur retournée. `json.loads()` accepte les espaces autour d’un nombre,
mais rejette un espace au milieu, comme dans `1 2`.

### 6.5 `choose_token(current_ids, allowed, model)`

**Entrées :** contexte, identifiants autorisés et SDK. **Sortie :** identifiant
ayant le meilleur score parmi les choix autorisés.

S’il n’existe qu’un choix, la structure impose ce token. Sinon, la fonction
demande les logits au SDK, garde les scores des identifiants autorisés et
choisit leur maximum avec NumPy. Les scores non finis sont exclus. Aucun choix
possible provoque une erreur explicite.

`function_select.py` importe cette fonction : sa signature est conservée.
Le nom et les arguments sont toujours sélectionnés à partir des scores du
modèle quand il existe plusieurs possibilités.

### 6.6 `string_completion(text, suffix)`

**Entrées :** texte candidat et structure attendue après la valeur.
**Sortie :** littéral JSON complet et suffixe restant, ou `None`.

Un seul token peut contenir `hello",`. Il faut conserver `hello` et reconnaître
la virgule déjà présente. `json.JSONDecoder().raw_decode(text)` donne la valeur
et l’emplacement de sa fin. La fonction exige une chaîne encodable en UTF-8,
puis vérifie que les caractères suivants correspondent au suffixe prévu.

Exemple : pour le candidat `"hello", "b` et le suffixe `, "b":`, la valeur
est `hello` et il reste à ajouter `":`. Le séparateur n’est pas dupliqué, et
le modèle ne peut pas inventer une autre clé. L’espace après la virgule est
facultatif.

### 6.7 `parse_scalar(text, kind)`

**Entrées :** littéral complet et type attendu. **Sortie :** valeur Python.

`json.loads()` interprète le texte, puis le nombre est converti en `float`
fini. Une chaîne reste un `str` et un booléen reste un `bool`. Une incohérence
provoque une erreur. Les espaces dans les chaînes sont conservés ; `\u00e9`
devient `é` et `\n` devient un retour à la ligne.

Les flottants gardent leurs limites de précision. Le programme ne remplace
jamais une valeur invalide par un zéro inventé.

### 6.8 `decode_parameter(...)`

**Entrées :** contexte, type, textes des tokens, SDK, suffixe attendu et limite
de génération. **Sortie :** un argument Python validé.

La fonction répète cinq opérations :

1. essayer chaque token à la suite du texte actuel ;
2. garder les préfixes autorisés ;
3. reconnaître aussi la fermeture d’une chaîne avec son suffixe éventuel ;
4. choisir le prochain token avec `choose_token()` ;
5. l’ajouter au contexte et poursuivre ou terminer.

Pour une chaîne, le guillemet initial est déjà dans le contexte : il ne faut
pas l’ajouter une seconde fois. Pour un nombre, le séparateur de fin devient
un choix possible seulement si `complete_number(text)` est vrai. Un booléen
se termine quand `true` ou `false` est complet.

La valeur passe ensuite par `parse_scalar()`. Seule la partie manquante du
suffixe est ajoutée. Au-delà de 256 tokens, une erreur signale une génération
inachevée ; la fonction ne renvoie pas un argument tronqué.

### 6.9 `extract_all_parameters(...)`

**Entrées :** définition sélectionnée, contexte, tokens et SDK.
**Sortie :** dictionnaire des arguments générés.

La fonction parcourt les arguments dans leur ordre de définition. Elle impose
les clés et la ponctuation avec `json.dumps()`, puis demande chaque valeur à
`decode_parameter()`. Une fonction sans argument renvoie `{}`.

La clé et le guillemet d’ouverture d’une chaîne sont encodés ensemble. Cela
évite de découper artificiellement cette structure en tokens peu naturels.
Le suffixe prépare la clé suivante ou ferme l’objet avec `}`.

## 7. `src/__main__.py` — le programme principal

### 7.1 `parse_args()`

Lit les options avec `argparse` :

| Option | Chemin par défaut |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calls.json` |

`--help` affiche leur usage. La fonction retourne un `Namespace` contenant ces
chemins.

### 7.2 `write_results(path, results)`

Crée le dossier parent si nécessaire, ouvre la sortie en UTF-8 avec `with`,
puis écrit le tableau par `json.dump()`.

- `indent=2` rend le résultat lisible ;
- `ensure_ascii=False` conserve les caractères Unicode ordinaires ;
- `allow_nan=False` interdit NaN et les infinis dans le JSON final.

Le fichier est fermé automatiquement. Il n’y a plus de fichier temporaire ni
de remplacement atomique. Si une écriture échoue, l’erreur remonte à `main()` ;
cette version simple ne garantit pas la conservation du précédent fichier
pendant une panne d’écriture.

### 7.3 `main()`

Suit le pipeline dans cet ordre :

1. lire les chemins ;
2. refuser que la sortie écrase l’un des fichiers d’entrée ;
3. charger les définitions et les questions avec le parseur ;
4. écrire `[]` sans modèle si les questions sont vides ;
5. charger Qwen une fois ;
6. lire et décoder le vocabulaire via le SDK ;
7. choisir une fonction pour chaque question ;
8. ajouter le début de l’objet `parameters` ;
9. extraire les arguments ;
10. appeler `validate_call()` pour les clés et types exacts ;
11. écrire les appels complets dans l’ordre initial.

Les erreurs sont interceptées au niveau principal et affichées sur `stderr`
avec `Error: ...`. Le programme sort avec un code non nul. Aucun appel inventé
n’est ajouté pour masquer un échec.

Le chronométrage et les messages de progression ont été retirés de cette version
pour alléger le programme. Le temps de vérification est mesuré de l’extérieur.

Le bloc `if __name__ == '__main__'` appelle `main()` lorsque le module est lancé
avec `python -m src`, pas quand il est importé.

## 8. Le SDK fourni

L’application appelle seulement son API publique. Torch, Transformers et
Hugging Face restent à l’intérieur du SDK, comme prévu par le sujet.
La classe d’origine `Small_LLM_Model` reste le wrapper fourni ; les classes
métier de `parser.py` utilisent Pydantic.

### `Small_LLM_Model.__init__(...)`

Charge le tokenizer et les poids de `Qwen/Qwen3-0.6B` par défaut. Le SDK conserve
ses paramètres d’origine de nom de modèle, périphérique et précision. L’application
n’ajoute pas d’option de choix de modèle ni de fonctionnalité multi-modèles.

Le SDK sélectionne MPS, sinon CUDA, sinon CPU, place le modèle sur le périphérique
et le met en mode évaluation. Les gradients des paramètres sont désactivés.
Le SDK actuel correspond au fichier du ZIP fourni. Les adaptations de typage
précédemment ajoutées au SDK ne sont plus présentes ; son code est exclu du lint.

### `encode(text)`

Convertit le texte en identifiants avec le tokenizer fourni et retourne un
tenseur contenant une séquence : `[[id_1, id_2, ...]]`.
L’application obtient la liste avec `.tolist()[0]`.

### `decode(ids)`

Convertit une liste d’identifiants ou un tenseur en texte, en ignorant les
marqueurs spéciaux à l’affichage. Cette méthode est utilisée directement pour
préparer le dictionnaire des tokens. Aucun `decode` maison n’est implémenté.

### `get_logits_from_input_ids(input_ids)`

Transforme le contexte complet en tenseur, appelle le modèle sans gradients et
renvoie les scores de la dernière position pour le prochain token.

L’appel est celui du SDK d’origine : `self._model(input_ids=input_tensor)`.
L’ajout `use_cache=False` n’est plus présent. L’application ne conserve aucun
état d’attention entre deux appels et fournit le contexte complet au SDK.

### `get_path_to_vocab_file()`

Renvoie le chemin du vocabulaire JSON, téléchargé ou déjà présent dans le cache
standard de fichiers Hugging Face. Ce cache de téléchargement appartient au SDK
fourni ; il est distinct du cache d’inférence bonus retiré.

### `get_path_to_merges_file()`

Renvoie le fichier des fusions du tokenizer. La méthode reste fournie par le SDK,
mais l’application ne l’utilise pas pour reconstruire un tokenizer.

### `get_path_to_tokenizer_file()`

Renvoie la configuration du tokenizer. Cette méthode fournie est également
conservée, sans être utilisée pour ajouter une autre implémentation de tokenizer.

## 9. Les commandes et l’environnement

`pyproject.toml` déclare le SDK local, NumPy et Pydantic. Les outils de développement
sont Flake8 et mypy. `uv.lock` fixe les versions résolues. `.venv` contient leur
installation locale.

```sh
uv sync
uv run python -m src
```

`uv sync` prépare l’environnement mais ne l’active pas dans le terminal.
`uv run` l’utilise pour la commande, même sans activation manuelle.

| Commande Make | Rôle |
|---|---|
| `make install` | Installer avec `uv sync`. |
| `make run` | Lancer l’application via `uv run python -m src`. |
| `make debug` | Lancer le même module sous `pdb`. |
| `make lint` | Exécuter Flake8 et les options mypy exigées sur `.`. |
| `make lint-strict` | Exécuter Flake8 puis `mypy . --strict`. |
| `make clean` | Supprimer caches de projet et sortie générée, sans supprimer `.venv` ni `.git`. |

`make lintr` est une faute de frappe. Il n’existe pas de cible `make test` dans
le Makefile actuel. Les anciens résultats de 22 tests d’une autre version ne
sont donc plus présentés comme la vérification de ce code.

Le SDK fourni est exclu de Flake8 et de mypy. Sa configuration mypy est :

```toml
[tool.mypy]
mypy_path = "$MYPY_CONFIG_FILE_DIR/llm_sdk"
exclude = ["^\\.venv/", "^llm_sdk/"]

[[tool.mypy.overrides]]
module = ["llm_sdk", "llm_sdk.*"]
follow_imports = "skip"
```

`mypy_path` permet de localiser le paquet. `exclude` retire le SDK de la recherche
des fichiers à contrôler ; `follow_imports = "skip"` empêche son analyse lorsqu’il
est importé depuis l’application. Ces réglages ne changent pas les imports ni
l’exécution. Les 9 fichiers Python de l’application et des expériences restent
contrôlés par les commandes `make lint` et `make lint-strict`.

## 10. Les expériences manuelles conservées

Aucune suite de tests complète nouvelle n’est ajoutée au rendu. Le dossier
`test/` contient les expériences qui existaient déjà :

- `test/test.py` : `speak_with_llm(prompt)` affiche des identifiants et logits,
  puis illustre un masquage de token. Ce n’est pas le pipeline à rendre.
- `test/tests.py` : `speak_with_llm(prompt)` affiche les identifiants de caractères
  JSON trouvés dans le vocabulaire.
- `test/testz.py` : `tchat_llm(prompt)` réalise une génération libre, arrêtée sur
  les tokens de fin. Elle ne garantit pas un appel JSON structuré.
- `test/__init__.py` marque le paquet ; il n’ajoute aucune fonctionnalité.

Leurs appels sont protégés par `if __name__ == '__main__'`. Les importer ne
charge donc pas automatiquement une expérience d’inférence.

Les vérifications ponctuelles réalisées pendant cette intervention sont placées
hors du dépôt, dans l’espace de travail de cette conversation. Vérifier le
mandatory ne consiste pas à ajouter un bonus de suite de tests au rendu.

## 11. Ce qui a été simplifié les 12 et 13 novembre

| Avant | Version actuelle |
|---|---|
| Cache des clés/valeurs d’attention dans le SDK | Aucun cache d’inférence conservé. |
| Conversion manuelle des symboles ByteLevel en octets | Appels au `decode()` public du SDK. |
| Classe `TokenVocabulary` et plusieurs groupes de tokens | Dictionnaire simple `dict[int, str]`. |
| Machine à états numérique en octets, puis expressions régulières difficiles à lire | `json.loads()` et vérification explicite des préfixes numériques. |
| Expression régulière pour les chaînes ouvertes | Boucle avec `escaped` et `missing_digits`, puis lecteur JSON pour la chaîne complète. |
| Fichier temporaire puis remplacement atomique | Écriture directe avec `with` et `json.dump()`. |
| Chronométrage et progression dans `main()` | Pipeline sans instrumentation intégrée. |
| Documentation d’une ancienne suite complète | Vérifications actuelles et expériences réellement présentes. |

Les expressions régulières `NUMBER`, `NUMBER_PREFIX` et `STRING_PREFIX` ont
été supprimées après ta remarque du 13 novembre. Les fonctions de vérification
utilisent maintenant du Python explicite. Le détail est dans la section 6.
Les longues docstrings du décodeur ont été remplacées par des descriptions
courtes ; les explications restent dans ce document.

Les noms de fonctions et valeurs viennent toujours du modèle. Il n’y a pas de
réponses mémorisées ou de règles de mots-clés pour les exemples du sujet.

## 12. Obligatoire ou bonus ?

| Fonctionnalité | Statut dans cette version |
|---|---|
| Qwen3-0.6B par défaut | Obligatoire, conservé. |
| Choix de fonction par le LLM | Obligatoire, conservé. |
| Contraintes JSON et types des arguments | Obligatoires, conservées. |
| Gestion normale des fichiers absents ou JSON malformés | Obligatoire, conservée. |
| Flake8, mypy, typage et docstrings | Obligatoires, conservés. |
| Cache d’inférence, batching | Bonus, non ajoutés. |
| Réécriture du tokenizer et API de tokenizer maison | Bonus, absentes. |
| Arguments complexes imbriqués | Bonus, absents. |
| Récupération avancée après erreur | Bonus, absente. |
| Suite complète ajoutée au rendu | Bonus, absente. |
| Visualisation de génération | Bonus, absente. |

La gestion des échappements, d’un nombre négatif ou de deux noms partageant un
préfixe n’est pas un bonus : elle sert à respecter la partie obligatoire sur des
entrées normales ou des cas limites.

## 13. Vérifications et limites

Le sujet exige un JSON valide et conforme au schéma, au moins 90 % de bons appels
sur les essais, et moins de cinq minutes pour le lot sur du matériel standard.

Vérifications du 13 novembre. Les lignes de lint ci-dessous ont été revérifiées
avec le SDK d’origine exclu. Les mesures de génération proviennent du lancement
antérieur à la restauration du SDK ; leur durée n’a pas été remesurée depuis.

| Contrôle | Résultat |
|---|---|
| `make lint` avec exclusion du SDK fourni | Réussi, code de sortie 0 ; 9 fichiers Python vérifiés par mypy. |
| `make lint-strict` avec exclusion du SDK fourni | Réussi, code de sortie 0 ; 9 fichiers Python vérifiés par mypy. |
| `uv run mypy src --strict` | Réussi, 5 fichiers. |
| `uv sync --locked` | Réussi. |
| Empreintes des deux fichiers protégés | Identiques à celles du début de la simplification. |
| Vérifications temporaires du décodeur | Nombres invalides, exposants, booléens, échappements, Unicode, espaces, tokens traversant une fermeture et limite de génération : réussies. |
| Vérifications temporaires de la ligne de commande | 8 cas réussis, dont fichiers absents, JSON malformé, entrées vides et protection des fichiers d’entrée. |
| Lancement réel de Qwen sur les 11 demandes fournies | Sortie 0, 267,29 secondes, soit environ 4 min 27 s. |
| JSON lisible et conforme au schéma | 11/11. |
| Bons noms de fonctions | 11/11. |
| Appels entièrement corrects sur ces 11 demandes | 10/11, soit 90,91 %. |

L’essai utilise le modèle déjà téléchargé ; ce temps ne comprend pas un premier
téléchargement. Les vérifications de la ligne de commande ont tourné en parallèle
pendant une partie de ce lancement. Cette durée décrit cette exécution sur cette
machine ; elle ne garantit pas la même durée sur tout matériel.

**Erreur réelle conservée dans le bilan :** pour remplacer les voyelles de
`Programming is fun` par des astérisques, Qwen produit un remplacement `" *"`
avec un espace initial, au lieu de `"*"`. Le JSON est valide, mais l’appel est
incorrect. La simplification n’a pas résolu cette erreur sémantique du modèle.

Trois demandes supplémentaires ont aussi été exécutées avec d’autres définitions
de fonctions, en 30,76 secondes. Elles couvrent un nombre négatif, des booléens,
le texte `café`, une fonction sans argument et une chaîne vide. Les trois sorties
respectent le schéma ; deux appels sont entièrement corrects. Pour la demande
exigeant un libellé vide `""`, Qwen produit `" "`, c’est-à-dire un espace.

Sur les deux lots réunis, cela donne donc **12 appels corrects sur 14, soit
85,71 %**, avec 14/14 sorties conformes au schéma. Le seuil de 90 % est atteint
sur les seuls exemples fournis, mais pas sur cet ensemble élargi. Il serait
incorrect de présenter la conformité complète au sujet comme acquise.

Les anciennes mesures de la version avec cache ne décrivent plus ce code.

La validité du schéma ne suffit pas à prouver le sens de la réponse : une valeur
peut avoir le bon type et être incorrecte. Les expressions régulières proposées
par le modèle peuvent aussi avoir plusieurs écritures équivalentes sur un exemple ;
la comparaison doit contrôler leur effet, la chaîne source et le remplacement.

Limites explicites : types `number`, `string`, `boolean` ; flottants finis ;
256 tokens par valeur ; pas d’objets ni de listes imbriquées ; certains tokens
Unicode incomplets exclus ; arrêt du lot avec erreur si une valeur ne peut pas
être terminée. Les résultats obtenus sur les exemples ne garantissent pas la
même précision sur des demandes inconnues.

## 14. Points à savoir expliquer à l’oral

1. Pourquoi ne retourne-t-on pas la somme de 2 et 3 ?
2. Quelle différence entre un JSON valide et un appel correct ?
3. À quoi sert le schéma Pydantic ?
4. Pourquoi `1.` est-il un préfixe valide, mais pas encore un nombre complet ?
5. Pourquoi un token contenant `",` doit-il être pris en compte ?
6. Pourquoi `True` ne doit-il pas être accepté comme nombre ?
7. Qui choisit une valeur quand plusieurs tokens sont autorisés ?
8. Pourquoi un token unique imposé par le schéma n’exige-t-il aucun choix ?
9. Quelle différence entre le cache de téléchargement du modèle et un cache
   d’inférence ?
10. Pourquoi utilise-t-on le SDK au lieu d’importer Transformers dans `src` ?

Pour suivre le code : `make debug`, puis des points d’arrêt dans
`select_valid_function()`, `decode_parameter()` et `validate_call()`.
