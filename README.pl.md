[![en](https://img.shields.io/badge/language-EN-blue.svg)](https://github.com/pzemla/Image-classification-using-CNN/blob/main/README.md)
# Klasyfikacja obrazów wykorzystując sieć CNN

**Zależności**

Python 3.9.13

matplotlib 3.8.3

notebook 7.1.2

numpy 1.24.1

pandas  2.2.1

scikit-learn 1.4.1.post1 Python 3.9.13

torch 2.2.1+cu118

**Jak uruchomić**
1. Pobierz dataset z https://drive.google.com/file/d/1ieJKBpl9ZQ2xQKjvkhwHkZrGwJLV1qpr/view?usp=sharing
2. Umieść foldery train i test_all w tym samym folderze co convolutional.ipynb
3. Uruchom convolutional.ipynb w Jupyter Notebook

## Przegląd

Celem tego projektu jest zbudowanie konwolucyjnej sieci neuronowej (CNN) do klasyfikowania obrazów 64x64x3 do jednej z 50 predefiniowanych klas. CNN jest zaimplementowany przy użyciu Pythona z biblioteką Pytorch. Model jest szkolony na oznaczonym zbiorze danych zawierającym obrazy z różnych kategorii, w celu dokładnego przewidzenia etykiet klas obrazów. Niska rozdzielczość obrazów jest spowodowana ograniczonym czasem i mocą obliczeniową możliwą do przeznaczenia na trenowanie sieci. Projekt ten służy jako ćwiczenie edukacyjne i praktyczne zastosowanie technik głębokiego uczenia się do zadań klasyfikacji obrazów.

Liczba obrazów na klasę w zbiorze treningowym wynosi zazwyczaj między 300 a 400 obrazów, z wyjątkiem dwóch klas, z których jedna ma około 200 obrazów, a druga poniżej 100. 

![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/af85d2ef-a690-48db-88c6-d627ccc7958d)


Poniżej są przykładowe zdjęcia obrazów z 3 różnych klas:

Baterie

![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/4bd53ef6-e0bc-4b7f-9e30-0b5a2174ccfe)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/085ab246-5d8b-4726-8e7b-6e70bb868ebc)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/59320a5b-1e69-4a05-92f6-afe9b5442d4e)

Żółwie 

![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/cdf39672-2bf8-475b-abb7-01140a03260f)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/5acd752a-0376-43f1-8ba0-cb1ccc47a90f)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/d6a8450b-59f2-41b1-b449-dc0627cc7280)

Ręczniki 

![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/5d383acb-46d5-4d2c-bf3c-5e55cbabda57)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/74cdf248-3258-42b0-b141-c7f092cdd2e4)
![image](https://github.com/pzemla/Image-classification-using-CNN/assets/135070990/8caf964f-1856-40b8-bac7-67119f141bf6)

# Struktura CNN
Konwolucyjna sieć neuronowa (CNN) to rodzaj sztucznej sieci neuronowej zaprojektowanej do analizy danych wizualnych, takich jak obrazy. Działa poprzez przepuszczanie obrazów wejściowych przez warstwy filtrów splotowych, które wyodrębniają takie cechy, jak krawędzie i tekstury. Funkcje te są następnie stopniowo łączone i analizowane w dodatkowych warstwach, umożliwiając sieci uczenie się złożonych wzorców i dokonywanie prognoz, takich jak rozpoznawanie obiektów lub klasyfikacja obrazu.

|Warstwa|Opis|Input|Output|
| ------------- | ------------- | ------------- | ------------- |
|Convolutional|Stride=1, kernel=3, padding=1|64x64x3|64x64x32|
|Batch normalization|Normalizacja outputu z warstwy konwolucyjnej|64x64x32|64x64x32|
|Relu|Funkcja aktywacji|64x64x32|64x64x32|
|Max pooling|Stride=2, kernel=2|64x64x32|33x33x32|
|Convolutional|Stride=1, kernel size=3, padding=1|33x33x32|33x33x64|
|Batch normalization|Normalizacja outputu z warstwy konwolucyjnej|33x33x64|33x33x64|
|Relu|Funkcja aktywacji|33x33x64|33x33x64|
|Max pooling|Stride=2, kernel=2|33x33x64|17x17x64|
|Convolutional|Stride=1, kernel size=3, padding=1|17x17x64|17x17x128|
|Batch normalization|Normalizacja outputu z warstwy konwolucyjnej|17x17x128|17x17x128|
|Relu|Funkcja aktywacji|17x17x128|17x17x128|
|Max pooling|Stride=2, kernel=2|17x17x128|9x9x128|
|Convolutional|Stride=1, kernel size=3, padding=1|9x9x128|9x9x256|
|Batch normalization|Normalizacja outputu z warstwy konwolucyjnej|9x9x256|9x9x256|
|Relu|Funkcja aktywacji|9x9x256|9x9x256|
|Max pooling|Stride=2, kernel=2|9x9x256|5x5x256|
|Convolutional|Stride=1, kernel size=3, padding=1|5x5x256|5x5x512|
|Batch normalization|Normalizacja outputu z warstwy konwolucyjnej|5x5x512|5x5x512|
|Relu|Funkcja aktywacji|5x5x512|5x5x512|
|Max pooling|Stride=2, kernel=2|5x5x512|3x3x512|
|Flatten|‘Spłaszczenie’ kształtu w wektor do warstwy liniowej|3x3x512|4608|
|Linear|Warstwa liniowa|4608|512|
|Dropout|Probability=0.6|512|512|
|Linear|Output 50 dla każdej z klas|512|50|


# Optymalizator i funkcja straty

Optymalizator – Adam (learning rate=0.0001)

Funkcja straty – Cross-entropy loss

Optymalizator Adam został wybrany ponieważ dynamicznie dostosowuje learning rate do każdego parametru podczas treningu, przez co nie trzeba dostosowywać malenia współczynnika uczenia (learning rate decay). Spośród innych optymalizatorów testowanych (Adagrad i RMSprop) zapewniał on najlepsze wyniki w datasecie testowym.

Funkcja straty entropii krzyżowej (cross-entropy loss) została wybrana ponieważ jej wynik może być interpretowany jako prawdopodobieństwo przynależności do każdej klasy, przez co często wykorzystuje się ją do modeli klasyfikujących.

# Rezultaty

Najlepsza osiągnięta dokładność wynosi 62%. Niektóre czynniki utrudniające osiągnięcie wyższej dokładności (z wyłączeniem ograniczeń dotyczących rozmiaru i architektury CNN) pochodzą ze zbioru danych. Zdjęcia mają niską rozdzielczość i jak widać na powyższych przykładowych zdjęciach, czasami sklasyfikowany obiekt jest ukryty w tle (osoba trzymająca baterie, które zajmują tylko niewielką część obrazu, żółw ukryty w wodzie, część ręcznika na wieszaku), oraz  mają stosunkowo niewielką liczba obrazów w każdej klasie. Biorąc pod uwagę te czynniki, dokładność 62% (w porównaniu z dokładnością około 2% w przypadku zgadywania losowego) wydaje się zadowalająca w przypadku stosunkowo prostej sieci CNN.
