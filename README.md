# MLP Example - Multi-Layer Perceptron Example

This is an simple C++11 implementation of a Multi-Layer Perceptron (MLP).

## Getting Started

The example provides a demonstration of use by the way of the iris dataset (subset). See iris.cpp in examples directory for the complete sample.

### Example

The exmaple provided works with a small subset of the iris dataset with two classes. The setup is shown below:

```
try {
    /**
     * Configure the MLP as 4 layers:
     */
    network<> nn({4,6,6,6,3});
    nn.alpha = 0.03;
    nn.on_epoch = [&]() {
        auto res = nn.test(data, labels);
        return res.accuracy >= 1.0f;
    };

    std::cout << "Untrained loss: " << nn.loss(data, labels) << "\n";

    nn.train(data, labels, 5000);

    std::cout << "Trained loss: " << nn.loss(data, labels) << "\n";

    std::cout << "Final Accuracy: " << nn.test(data, labels).accuracy * 100.0f << "%\n";

}catch(mlp_error& err) {
    std::cout << "Error occured: " << err.why << "\n\n";
}
```

The sample can be run with:

```
make && make test
```

For which the output is:

```
$ make test
./example/iris.out ./example/iris.csv
Untrained loss: 1.646
Trained loss: 0.0320878
Final Accuracy: 99.3333%

```

### Installing

No installation is necessary, headers are located in mlp directory. The example (iris.cpp) uses headers only.

## Authors

* **Omar Thor** - *MLP Example* - [thorigin](https://github.com/thorigin/mlp-example)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

