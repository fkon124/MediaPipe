# Prepoznavanje gesti ruke pomoću MediaPipe-a

Sustav za prepoznavanje gesti ruke temeljen na **MediaPipe Hands** biblioteci i **Random Forest** klasifikatoru iz `scikit-learn` paketa.  
Projekt omogućuje prikupljanje vlastitog skupa podataka kamerom, treniranje modela i prepoznavanje gesti u stvarnom vremenu.

---

## Pregled projekta

MediaPipe prepoznaje 21 ključnu točku šake (*landmark*), svaka s koordinatama `(x, y, z)` — ukupno **63 numeričke značajke**. Landmark vektor se normalizira (translacija na zapešće, skaliranje prema udaljenosti točke 9) kako bi model bio neovisan o poziciji i veličini ruke u kadru.

Projekt se sastoji od dvije skripte:

- `train_gestures.py` — interaktivno prikupljanje uzoraka kamerom i treniranje modela
- `test_gestures.py` — prepoznavanje gesti u stvarnom vremenu ili na pojedinačnoj slici

---

## Podržane geste

| Ključ | Gesta | Opis |
|-------|-------|------|
| `1` | `none` | nema geste |
| `2` | `ok` | OK znak |
| `3` | `thumbs_up` | palac gore |
| `4` | `thumbs_down` | palac dolje |
| `5` | `peace` | znak mira |
| `6` | `pointing` | pokazivanje prstom |
| `7` | `love` | I love you |
| `8` | `rock` | rock znak |
| `9` | `mobitel` | palac + mali prst |

---

## Tehnologije

- Python 3.10+
- OpenCV
- MediaPipe `0.10.14`
- NumPy
- scikit-learn
- joblib

---

## Instalacija

### 1. Kloniranje repozitorija

```bash
git clone https://github.com/username/ime-repozitorija.git
cd ime-repozitorija
```

### 2. Instalacija ovisnosti

```bash
pip install opencv-python mediapipe==0.10.14 numpy scikit-learn joblib
```

Ili putem `requirements.txt`:

```txt
opencv-python
mediapipe==0.10.14
numpy
scikit-learn
joblib
```

```bash
pip install -r requirements.txt
```

> **Napomena:** Koristi točno verziju `mediapipe==0.10.14`. Novije verzije ne podržavaju `mp.solutions` sučelje koje projekt koristi.

---

## Struktura projekta

```text
.
├── train_gestures.py
├── test_gestures.py
├── dataset/
│   ├── none/
│   ├── ok/
│   ├── thumbs_up/
│   ├── thumbs_down/
│   ├── peace/
│   ├── pointing/
│   ├── love/
│   ├── rock/
│   └── mobitel/
├── exported_model/
│   ├── gesture_classifier.pkl
│   └── gesture_labels.txt
└── README.md
```

Mape u `dataset/` automatski se kreiraju pri prvom pokretanju `train_gestures.py`. Svaki uzorak sprema se kao `.npy` datoteka (vektor oblika `(63,)`). Ako mapa sadrži `.npy` datoteke, one se koriste za trening; inače se automatski obrađuju slikovne datoteke (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).

---

## Korištenje

### 1. Prikupljanje podataka i treniranje modela

```bash
python train_gestures.py
```

Otvorit će se prikaz web kamere. Odaberi gestu tipkama i pritisni `S` za snimanje uzorka.

#### Kontrole

| Tipka | Funkcija |
|-------|----------|
| `1` – `9` | odabir geste (vidi tablicu gesta) |
| `S` | spremi trenutni uzorak (min. razmak: 0.3 s) |
| `Q` ili `ESC` | izlaz i pokretanje treninga |

Na ekranu se prikazuju: naziv aktivne geste, oznaka ruke (lijeva/desna) i broj dosad snimljenih uzoraka za odabranu gestu.

#### Automatski trening

Nakon zatvaranja kamere skripta automatski:
1. učitava sve `.npy` uzorke iz `dataset/`
2. dijeli podatke na trening (80 %) i validaciju (20 %)
3. trenira `RandomForestClassifier` (`n_estimators=300`, `max_depth=18`, `min_samples_leaf=3`)
4. ispisuje `classification_report` za validacijski skup
5. sprema model u `exported_model/gesture_classifier.pkl` i labele u `exported_model/gesture_labels.txt`

#### Preporuka za kvalitetu modela

- minimalno 50–100 uzoraka po gesti
- snimaj pod različitim kutovima i udaljenostima ruke
- variiraj osvjetljenje i pozadinu

---

### 2. Testiranje modela uživo

```bash
python test_gestures.py
```

Sustav otvara kameru (zadana: `CAMERA_ID = 0`), za svaki frame:
1. detektira ruku i izvlači landmarke
2. normalizira vektor
3. klasificira gestu i prikazuje naziv, pouzdanost (0–1) i oznaku ruke

#### Testiranje na jednoj slici

Funkcija `run_image` dostupna je programski:

```python
from test_gestures import load_model, run_image

clf, label_names = load_model("exported_model")
run_image(clf, label_names, "moja_slika.jpg")
```

#### Kontrole

| Tipka | Funkcija |
|-------|----------|
| `Q` ili `ESC` | izlaz |

---

## Kako projekt radi

### Normalizacija landmarka

```python
pts = pts - pts[0]          # translacija: zapešće (točka 0) u ishodište
scale = np.linalg.norm(pts[9])  # skaliranje: udaljenost sredine dlana
pts = pts / scale
```

Normalizacija čini model neosjetljivim na poziciju ruke u kadru i na udaljenost od kamere.

### Parametri modela (promjenjivi u `train_gestures.py`)

| Varijabla | Zadana vrijednost | Opis |
|-----------|------------------|------|
| `N_ESTIMATORS` | 300 | broj stabala |
| `MAX_DEPTH` | 18 | maksimalna dubina stabla |
| `MIN_SAMPLES_LEAF` | 3 | minimalni uzorci po listu |
| `VAL_FRACTION` | 0.2 | udio validacijskog skupa |
| `CAMERA_ID` | 0 | indeks kamere |
| `SAVE_INTERVAL` | 0.3 | minimalni razmak između snimanja (sekunde) |

---

## Dodavanje novih gesti

1. Dodaj naziv u `GESTURE_FOLDERS` (obje skripte)
2. Dodaj prijevod u `LABEL_HR` (obje skripte)
3. Dodaj tipku u `KEY_TO_GESTURE` (`train_gestures.py`)
4. Ponovo prikupi uzorke i treniraj model

---

## Rješavanje problema

### MediaPipe ne detektira ruku

- Provjeri osvjetljenje i vidljivost cijele šake
- Po potrebi smanji `min_detection_confidence` u kodu

### `mediapipe.solutions` ne postoji

Koristi točnu verziju:

```bash
pip install mediapipe==0.10.14
```

### Model ne postoji pri pokretanju `test_gestures.py`

Prethodno pokreni `train_gestures.py` kako bi se generirali `exported_model/gesture_classifier.pkl` i `exported_model/gesture_labels.txt`.

### Niska točnost modela

- Prikupi više uzoraka po gesti
- Snimi pod različitim uvjetima (kut, osvjetljenje, udaljenost)
- Povećaj `N_ESTIMATORS`

---

## Licenca

Projekt je open-source i namijenjen u svrhe završnog rada.
