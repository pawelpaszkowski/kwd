import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

housingData = datasets.load_boston()

#analiza danych wejsciowych
plt.figure(1)
housingData_X = housingData.data[:, 0]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości nieruchomości od współczynnika przestępczości')
plt.xlabel('Współczynniik przestępczości')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

plt.figure(2)
housingData_X = housingData.data[:, 1]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości nieruchomości od części gruntów na działki')
plt.xlabel('Grunty przeznaczone na działki')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

plt.figure(3)
housingData_X = housingData.data[:, 2]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości nieruchomości od części powierzchni bez handlu')
plt.xlabel('Odsetek powierzchni niedotkniętych handlem')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

plt.figure(4)
housingData_X = housingData.data[:, 3]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości nieruchomości od ograniczeń drogi w postaci rzeki')
plt.xlabel('Ograniczenie w postaci rzeki')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

plt.figure(5)
housingData_X = housingData.data[:, 4]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości nieruchomości od stężenia tlenku azotu')
plt.xlabel('Stężenie tlenku azotu')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()
print((housingData_X))

plt.figure(6)
housingData_X = housingData.data[:, 5]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości nieruchomości od średniej liczby pokoi w mieszkaniu')
plt.xlabel('Średnia liczba pokoi na mieszkanie')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()
print((housingData_X))

plt.figure(7)
housingData_X = housingData.data[:, 6]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości obiektów od ich części zajętej przez właściciela')
plt.xlabel('Odsetek obiektów zajęta przez właściciela')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

plt.figure(8)
housingData_X = housingData.data[:, 7]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości obiektów od odległości do centrów pracy')
plt.xlabel('Odległość od obiektów zatrudnienia')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

plt.figure(9)
housingData_X = housingData.data[:, 8]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości obiektów od wskaźnika dostępności do autostrad')
plt.xlabel('Wskaźnik dostępności do autostrad')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

plt.figure(10)
housingData_X = housingData.data[:, 9]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości obiektów od wartości podatku od nieruchomości')
plt.xlabel('Wartość podatku od nieruchomości')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

plt.figure(11)
housingData_X = housingData.data[:, 10]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości obiektów od współczynnika uczniów do nauczycieli')
plt.xlabel('Współczynnik uczniów do nauczycieli')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

plt.figure(12)
housingData_X = housingData.data[:, 11]
housingData_X = housingData_X[:, np.newaxis]
plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości obiektów od współczynnika liczby Afroamerykanów')
plt.xlabel('Współczynnik liczby Afroamerykanów')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

plt.figure(13)

print(housingData)
housingData_X = housingData.data[:, 12]
housingData_X = housingData_X[:, np.newaxis]

plt.subplot('111')
plt.scatter(housingData_X,housingData.target)
plt.title('Zależność wartości obiektów od części mieszkańców o niższym statusie')
plt.xlabel('Procent mieszkańców o niższym statusie społecznym')
plt.ylabel('Wartość nieruchomości (w 1000$)')
plt.grid()
plt.show()

print("Probki: ")
sample1=housingData.data[0,:]
sample2=housingData.data[99,:]
sample3=housingData.data[505,:]
print(sample1)
print(sample2)
print(sample3)
print("Wartość nieruchomości")
print(housingData.target[0])
print(housingData.target[99])
print(housingData.target[505])