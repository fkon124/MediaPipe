# Prepoznavanje geste ruke pomoću MediaPipe-a

Sustav za prepoznavanje gesti ruke temeljen na **MediaPipe Hands** biblioteci i **Random Forest** klasifikatoru iz `scikit-learn` paketa.  
Projekt omogućuje prikupljanje vlastitog skupa podataka, treniranje modela i prepoznavanje gesti u stvarnom vremenu pomoću web kamere.

---

## Pregled projekta

Ovaj projekt koristi računalni vid za detekciju i klasifikaciju gesti ruke.  
MediaPipe prepoznaje 21 ključnu točku šake (*landmark*), dok model strojnog učenja klasificira gestu na temelju tih podataka.

Projekt se sastoji od dvije glavne skripte:

- `train_gestures.py` → prikupljanje podataka i treniranje modela
- `test_gestures.py` → testiranje modela uživo preko web kamere

---

## Funkcionalnosti

- Detekcija ruke i izdvajanje 21 landmark točke
- Pretvaranje landmark-ova u 63 numeričke značajke `(x, y, z)`
- Interaktivno prikupljanje podataka pomoću tipkovnice
- Treniranje vlastitog Random Forest modela
- Spremanje istreniranog modela pomoću `joblib`
- Prepoznavanje gesti u stvarnom vremenu
- Vizualni prikaz landmark točaka i predviđene geste

---

## Podržane geste

| Gesta | Opis |
|---|---|
| `none` | nema geste |
| `ok` | OK znak |
| `thumbs_up` | palac gore |
| `thumbs_down` | palac dolje |
| `peace` | znak mira |
| `pointing` | pokazivanje prstom |
| `love` | I love you |
| `rock` | rock znak |
| `mobitel` | palac + mali prst |

---

## Tehnologije

- Python 3.10.x
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

Ili pomoću `requirements.txt` datoteke:

```txt
opencv-python
mediapipe==0.10.14
numpy
scikit-learn
joblib
```

Instalacija:

```bash
pip install -r requirements.txt
```

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
│   └── ...
├── exported_model/
│   ├── gesture_classifier.pkl
│   └── gesture_labels.txt
└── README.md
```

---

# Korištenje

## 1. Prikupljanje podataka i treniranje modela

Pokreni:

```bash
python train_gestures.py
```

Otvorit će se prikaz web kamere.

### Kontrole

| Tipka | Funkcija |
|---|---|
| `1 - 9` | odabir geste |
| `S` | spremanje trenutnog uzorka |
| `Q` ili `ESC` | izlaz |

Nakon završetka prikupljanja podataka skripta automatski:

- učitava spremljene uzorke
- dijeli podatke na trening i validaciju
- trenira Random Forest klasifikator
- sprema model u `exported_model/`

### Preporuka

Za bolje rezultate preporučuje se:

- minimalno 50–100 uzoraka po gesti
- različiti kutovi ruke
- različita udaljenost od kamere
- različito osvjetljenje i pozadina

---

## 2. Testiranje modela uživo

Pokreni:

```bash
python test_gestures.py
```

Sustav će:

- detektirati ruku
- prikazati landmark točke
- prikazati predviđenu gestu i razinu sigurnosti modela

Izlaz:

```bash
Q ili ESC → izlaz iz programa
```

---

## 3. Testiranje na slici

U skripti `test_gestures.py` nalazi se funkcija `run_image()` koja omogućuje testiranje modela na jednoj slici umjesto web kamere.

Potrebno je samo promijeniti glavni dio skripte (`main`) i pozvati tu funkciju.

---

# Kako projekt radi

## 1. Detekcija ruke

MediaPipe Hands detektira 21 landmark točku na šaci.

Svaka točka sadrži:

- `x`
- `y`
- `z`

koordinate.

Ukupno:

```text
21 × 3 = 63 značajke
```

---

## 2. Prikupljanje podataka

Svaki uzorak sprema se kao `.npy` datoteka unutar odgovarajuće mape geste.

Primjer:

```text
dataset/peace/sample_01.npy
```

---

## 3. Treniranje modela

Za klasifikaciju se koristi:

```python
RandomForestClassifier
```

Model uči prepoznavati obrasce landmark točaka za svaku gestu.

---

## 4. Prepoznavanje u stvarnom vremenu

Za svaki frame web kamere:

1. MediaPipe izdvoji landmark točke
2. Landmark podaci pretvaraju se u feature vektor
3. Model predviđa gestu
4. Rezultat se prikazuje na ekranu

---

# Prilagodba projekta

## Dodavanje novih gesti

Potrebno je:

1. dodati novu mapu u `dataset/`
2. ažurirati `GESTURE_FOLDERS`
3. ažurirati `LABEL_HR`

---

## Promjena parametara modela

U `train_gestures.py` moguće je promijeniti:

```python
N_ESTIMATORS
VAL_FRACTION
```

---

## Promjena kamere

Promijeni:

```python
CAMERA_ID = 0
```

Ako koristiš više kamera:

- `0` → zadana kamera
- `1` → druga kamera
- `2` → treća kamera

---

# Rješavanje problema

## MediaPipe ne detektira ruku

Provjeri:

- da je ruka dobro osvijetljena
- da je cijela šaka vidljiva
- da pozadina nije previše kompleksna

Po potrebi povećaj:

```python
min_detection_confidence
```

---

## Greška: `mediapipe.solutions` ne postoji

Koristi točnu verziju:

```bash
pip install mediapipe==0.10.14
```

---

## Model ima nisku točnost

Pokušaj:

- prikupiti više podataka
- koristiti različite pozicije ruke
- povećati `n_estimators`
- testirati drugi klasifikator

---

# Moguća proširenja projekta

- Dodavanje više gesti
- Korištenje neuronskih mreža
- Spremanje podataka u CSV bazu
- Integracija s aplikacijama ili igrama
- Upravljanje računalom pomoću gesti

---

# Zahvale

- MediaPipe — detekcija landmark točaka ruke
- scikit-learn — Random Forest klasifikator
- OpenCV — obrada slike i prikaz web kamere

---

# Licenca

Projekt je dostupan pod MIT licencom.
