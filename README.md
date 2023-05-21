# Network-CV

About
-----
* The implementation of the UI introduced in the paper below:
  
  * Lu et al., "Visual Analytics of Multivariate Networks with Representation Learning and Composite Variable Construction." Visual Informatics, forthcoming. [arXiv](https://arxiv.org/abs/2303.09590)

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
### Data Preparation (WIP)
* Sample data will be provided soon.

### Server Setup

* Download and move to this repository. Then,

    `pip3 install -r requirements.txt`

* Run websocket server:

    `python3 ws_server.py` or `python ws_server.py`

* Run http server. For example, open a new tab in Terminal and then:

    `python3 -m http.server` or  `python -m http.server`

### Client Setup

* Access to the url setup in the http server. For example, if you set an http server with the above command. You can acess with: `http://localhost:8000/`


How to cite
-----
* If you use the workflow to analyze multivariate networks in your publication, please cite:

  * H.-Y. Lu, T. Fujiwara, M.-Y. Chang, Y.-c. Fu, A. Ynnerman, and K.-L. Ma. “Visual Analytics of Multivariate Networks with Representation Learning and Composite Variable Construction.” Visual Informatics, forthcoming.
