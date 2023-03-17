# Absyst: Generalising Patterns in Complex Datasets

Before I begin listing out all the processes of development and explaining the workings of the algorithm, I want to clarify that this algorithm is not recommended for use. This algorithm was developed without an actual mathematical foundation and was entirely programmed in Python. Therefore, the algorithm is quite inconsistent and also quite inefficient. I’ve had times when this algorithm would outperform other algorithms like K-Nearest Neighbours, or even Decision Trees, but I’ve also had times when the accuracy was around 40 in a completely reasonable classification task. Anyway, I’ve warned you. If you really care about your projects, don’t use this classifier. If you want just to poke around though, go ahead.

Complex datasets refer to large or sophisticated datasets, such as high-resolution image data, high-frequency financial data, or genomic data. These are known to be ‘complex’ as they hold huge numbers of variables and a high level of complexity and sophistication. Due to this, many classification algorithms struggle with dealing with such datasets, as high complexity may feed unnecessary or misleading patterns to the algorithms, leading to lower accuracy. I started working on Absyst hoping to find whether a classification algorithm could be partially immune to such noises. 

## Data Abstraction
During a training phase, complex data with a large number of variables are fed into the algorithm. However, this would mean that there is a lot of variation in variables. Take images for example. A large dataset composed of high-resolution images with colours would mean that each image would be vastly different from the others. To minimise the complexity of variables, the classifier uses a process I call abstraction.



When the data is fed in the form of an array, the classifier generates a difference matrix of the input, which is a matrix that outlines the differences between elements in the input matrix. By using this process, the algorithm can ensure that differences in colour won’t affect the classification process. With a difference matrix, the algorithm can solely focus on the features, and won’t be hindered by unnecessary prevalent features, such as a background colour or the colours of an object.

However, this entirely depends on the use case. For example, if a user is attempting to classify images using colour, disabling this option would be beneficial. Because of this, I’ve made this an accessible parameter called abstraction_depth, with its default value being 1. Abstraction depth can actually go above 1 (meaning it will generate a difference matrix of the input), but making a difference matrix of a difference matrix… and so on does not affect the classification process. It is still an option though.

Data Generalisation
For a default state, once the individual inputs have been converted to their respective difference matrices, the classifier will compress all these inputs into one and create a single fusion matrix. This is done by finding the average of all the elements per index. Finding an average matrix of all input matrices makes the comparing process much more efficient.




