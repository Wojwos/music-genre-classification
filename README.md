# music-genre-classification

### Wymagania programowe
Aby móc korzystać z tej pracy należy posiadać następujące oprogramowanie i narzędzia:
- Python 3.8.3,
- Pakiety python:
   - NumPy,
   - Pandas,
   - Librosa,
   - Scikit-learn,
   - Tensorflow,
   - Matplotlib,
   - OpenCV,
- JupyterLab.

### Sposób instalacji
W celu przygotowania kompletnego środowiska należy zrealizować następujące kroki:

a) sklonować repozytorium i umieścić je w nowym katalogu,

b) zainstalować oprogramowanie Anaconda, oprogramowanie dostępne jest pod adresem www.anaconda.com,

c) zainstalować JupyterLab, w programie Anaconda Navigator przycisk install,

d) uruchomić Anaconda Prompt:
- stworzyć nowe środowisko:
conda create --name music-genre-classification
- aktywować środowisko:
activate music-genre-classification

e) zainstalować pakiety korzystając z narzędzia conda:
- conda install python
- conda install jupyterlab
- conda install pandas
- conda install matplotlib
- conda install scikit-learn
- conda install tensorflow
- conda install -c conda-forge librosa

f) aby móc zainstalować inne pakiety spoza repozytorium conda należy zainstalować pip:
- conda install pip
- pip install opencv-contrib-python

### Sposób obsługi
**Przygotowanie środowiska**

Przed rozpoczęciem pracy należy przygotować środowisko, instalując wcześniej wszystkie wymagane biblioteki wraz z narzędziem JupyterLab. W folderze z plikami MusicGenreClassification.ipynb oraz MusicGenreClassificationTool.py utworzyć nowy folder i nazwać go “Samples”.

**Przygotowanie danych**

Aby móc korzystać z modułu pierwszym krokiem jest odpowiednie przygotowanie danych. Powinny być to pliki muzyczne zapisane w formatach takich, jak np. mp3 lub wav. Wszystkie pliki powinny być posortowane w folderach według ich gatunku. Każdy folder powinien być nazwany gatunkiem utworów znajdujących się wewnątrz. Wszystkie foldery z muzyką powinny zostać umieszczone w katalogu “Samples”. Tak przygotowane dane są gotowe do użycia.

```bash
├── Samples/
   ├── blues
   ├── classical
   ├── country
   .
   .
   .
   ├── rock
```

**Uruchamianie**

Po przygotowaniu środowiska i danych, należy uruchomić JupyterLab i otworzyć notatnik MusicGenreClassification.ipynb. Kolejnym krokiem jest uruchomienie wybranych funkcji, z których należy skorzystać w zależności od danego podejścia. Jednakże przed przystąpieniem do testowania własnych lub gotowych architektur sieci głębokich należy uruchomić wybrane funkcje przygotowujące dane do trenowania modeli. Funkcje, o których mowa znajdują się w sekcji “Data preparation”. Każda z nich przetwarza wszystkie utwory z katalogu “Samples” i tworzy nowe pliki oraz foldery w katalogu, w którym uruchomiony został projekt. W następnym kroku należy wybrać jakie realizowane podejście, w tym przypadku konieczne jest przejście do odpowiedniej sekcji. Do wyboru są trzy różne podejścia takie jak:
- spectrograms (image),
- spectrograms (array),
- features.
W każdej sekcji znajdują się trzy podsekcje dotyczące:
- wczytywania danych,
- budowania i trenowania modeli,
- testowania modeli.
Nie ma potrzeby nanoszenia zmian przy wczytywaniu danych, natomiast w przypadku podsekcji dotyczącej tworzenia modeli jest ona tylko i wyłącznie wskazówką gdzie oraz w jaki sposób można stworzyć model. Można go zastąpić dowolną architekturą dopasowaną do kształtu danych, a następnie przetestować model w taki sam sposób jak w gotowych rozwiązaniach.

**Obsługa notatnika Jupyter**
- **Importowanie bibliotek**

Pierwszą rzeczą jaką należy wykonać po uruchomieniu notatnika Jupyter to zaimportowanie danych. Wszystkie biblioteki konieczne do poprawnego działania narzędzia są zapisane w pierwszej komórce, dlatego należy jedynie ją uruchomić. W tym miejscu można również dołączyć inne biblioteki wykorzystywane przy własnych architekturach.

- **Przygotowywanie danych**

Następnym krokiem jest przygotowanie danych. Sekcja “Data preparation” zawiera w sobie 3 komórki każda z nich przetwarza dane zapisane w katalogu “Samples”. W przypadku wyboru jednego rozwiązania wystarczy uruchomić tylko funkcję związaną z danym podejściem.

- **Wczytywanie danych**

Przed wczytaniem danych ważne jest wybranie jednego z 3 podejść. W tym celu należy przejść do komórek znajdujących się w odpowiedniej sekcji. Istnieją 3 sekcje nazwane kolejno:
- spectrograms (image),
- spectrograms (array),
- features.
Po przejściu do odpowiednich komórek należy uruchomić kod znajdujący się pod oznaczeniem “Load the data:”.

- **Budowanie modeli**

Kolejnym elementem notatnika jest budowa modeli. W miejscu pod oznaczeniem “Build the model:” można wykonać zapisany kod. Natomiast w celu skorzystania z własnych architektur należy zmodyfikować kod. Koniecznym jest zapisanie historii z uczenia modeli, ponieważ jest ona wymagana w kolejnym punkcie. Konieczne jest aby historia zawierała odpowiednio nazwane takie jak: “loss”, “accuracy”, “val_loss”, “val_accuracy”.

- **Testowanie modeli**

Aby przetestować utworzony model należy uruchomić wszystkie komórki znajdujące się w podsekcji nazwanej “Test the model:”.
