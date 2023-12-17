# Trabalho-Programa-o
Trabalho de programação. Tema escolhido: KNN (K Nearest Neighbors). Mariana Pires e José Cunha 


#Welcome to KNN code. In here you'll find a simplified code that will calculate the KNN distance with the users input of the number of neighbours and with or without normalized data. 

#Example of usage of the code: 
# Create synthetic data
np.random.seed(42)
X_train = np.random.rand(100, 2) * 10
y_train = np.random.randint(0, 2, size=100)

# User inputs
Ks = int(input("Enter the number of neighbors (K): "))
normalize_option = input("Normalize data? (yes/no): ").lower() == 'yes'

# Create and fit KNN model based on user inputs
knn = KNN(K=Ks, normalize=normalize_option)
knn.fit(X_train, y_train)

# Generate test data (random points for prediction)
X_test = np.random.rand(10, 2) * 10

# Predict using the fitted model
predicted_labels = knn.predict(X_test)
print("Predicted labels:", predicted_labels)

# Plotting the training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training Data')
plt.title('Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the test data and predicted labels
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted_labels, cmap='viridis', label='Predicted Labels')
plt.title('Predicted Labels for Test Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Predicted Class')
plt.legend()
plt.grid(True)
plt.show()

#Plots: 
![Plor for predicted labels  png](https://github.com/MarianaPires93/Trabalho-Programa-o/assets/154060433/d193ca5a-3dd2-4857-9391-a461cac57dbe)

![Plot for training data](https://github.com/MarianaPires93/Trabalho-Programa-o/assets/154060433/a558f96c-3fae-4275-a9ec-17b3fee64849)

