# Network-CV

About
-----
* The implementation of the UI introduced in the paper below:
  
  * Lu et al., "Visual Analytics of Multivariate Networks with Representation Learning and Composite Variable Construction." arXiv:2303.09590, 2023. [arXiv](https://arxiv.org/abs/2303.09590)

* [demo video](https://youtu.be/Vro6uFGYBho)
* [Two-class density scatterplot](https://github.com/takanori-fujiwara/two-class-density-scatterplot) implementation

Requirements
-----
* Server side
  * Python3 (latest)
  * HTTP Server
* Client side
  * Browser supporting JavaScript ES2015(ES6).
  * Internet connection (to access D3 library)
* Note: Tested on macOS Ventura with Google Chrome.

Setup
-----

### Server Setup

* Download and move to this repository. Then,

    `pip3 install -r requirements.txt`

* Run websocket server:

    `python3 ws_server.py` or `python ws_server.py`

* Run http server. For example, open a new tab in Terminal and then:

    `python3 -m http.server` or  `python -m http.server`

### Client Setup

* Access to the url setup in the http server. For example, if you set an http server with the above command. You can acess with: `http://localhost:8000/`

### Data Preparation (when analyzing your own dataset)

* Refer to `./data/data.json` as a sample dataset.

* `./data/data_preparation.py` provides an example code to prepare a dataset using network representation learning, neural network classification, linear discriminant analysis, and SHAP. 
  * The sample dataset is made from US Faculty Hiring Netoworks: https://github.com/LarremoreLab/us-faculty-hiring-networks

* Detail information of data structure:
  * nodes: list of x, y-coordinates for Network Layout and class labels (0: Class 0, 1: Class 1, -1: Others)
    * e.g., "nodes": [{"x": 0.5, "y": 1.0, "label": 0}, {"x": 0.1, "y": 0.2, "label": 1}, {"x": 0.4, "y": 0.2, "label": -1}]
  
  * links: list of source and target node indices
    * e.g., "links": [{"source": 0, "target": 1}, {"source": 0, "target": 2}]

  * processed_node_indices: list of indices of nodes that are used for classification (i.e., Class 0 and Class 1 nodes) corresponding to the order of the feature matrix rows, representative values, and shap values below.
    * e.g., "processed_node_indices": [0, 1]

  * attr_values: a feature matrix used for clasisfication (shape: n_processed_node_indices x n_attributes)
    * e.g., "attr_values": [[0.5, 0.3], [0.4, 0.1]]
    
  * attr_names: list of attribute/feature names corresponding to the feature matrix columns.
    * e.g., "attr_names": ["degree", "eigenvector", "betweenness", "age"]

  * representative_values: 1D representative values for Class 0 and 1 nodes (i.e., x-coodinates for Representation Assessment)
    * e.g., "representative_values": [0.1, 0.2]

  * shap_values: disctionary of shap values for top-k contributed attributes (i.e., row names for Attribute Contributions and x-coorinates for each of the rows)
    * e.g., "shap_values": {"degree": [-0.1, 0.3], "age": [0.4, -0.5]}

  * classification_info: classification information displayed in Auxiliary Information
    * e.g., "classification_info" {
      "target_variable": "Rank",
      "class_0": "Top 25%",
      "class_1": "Bottom 25%",
      "accuracy": 0.95
    }

How to cite
-----
* If you use the workflow to analyze multivariate networks in your publication, please cite:

  * H.-Y. Lu, T. Fujiwara, M.-Y. Chang, Y.-c. Fu, A. Ynnerman, and K.-L. Ma. “Visual Analytics of Multivariate Networks with Representation Learning and Composite Variable Construction.” arXiv:2303.09590, 2023.
