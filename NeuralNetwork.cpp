// includes
#include "NeuralNetwork.hpp"
// Needed for std::find
#include <algorithm>
using namespace std;



// NeuralNetwork -----------------------------------------------------------------------------------------------------------------------------------

// STUDENT TODO: IMPLEMENT
void NeuralNetwork::eval() {
    evaluating = true;
}

// STUDENT TODO: IMPLEMENT
void NeuralNetwork::train() {
    evaluating = false;
    batchSize = 0;
}

// STUDENT TODO: IMPLEMENT
void NeuralNetwork::setLearningRate(double lr) {
    learningRate = lr;
}

// STUDENT TODO: IMPLEMENT
void NeuralNetwork::setInputNodeIds(std::vector<int> inputNodeIds) {
    this->inputNodeIds = inputNodeIds;
}

void NeuralNetwork::setOutputNodeIds(std::vector<int> outputNodeIds) {
    this->outputNodeIds = outputNodeIds;
}

// STUDENT TODO: IMPLEMENT
vector<int> NeuralNetwork::getInputNodeIds() const {
    return inputNodeIds; //stub
}

// STUDENT TODO: IMPLEMENT
vector<int> NeuralNetwork::getOutputNodeIds() const {
    return outputNodeIds; //stub
}

// STUDENT TODO: IMPLEMENT
vector<double> NeuralNetwork::predict(DataInstance instance) {

    flush();

    vector<double> input = instance.x;

    if (input.size() != inputNodeIds.size()) {
        cerr << "input size mismatch." << endl;
        cerr << "\tNeuralNet expected input size: " << inputNodeIds.size() << endl;
        cerr << "\tBut got: " << input.size() << endl;
        return vector<double>();
    }

    for (int i = 0; i < inputNodeIds.size(); i++) {
        int inputId = inputNodeIds[i];
        NodeInfo* inputNode = nodes.at(inputId);
        inputNode->preActivationValue = input.at(i);
        inputNode->activate();
    }

    vector<int> inDegrees(nodes.size(), 0);
    for (int v = 0; v < adjacencyList.size(); v++) {
        for (const auto& [u, conn] : adjacencyList[v]) {
            inDegrees[u]++;
        }
    }
    queue<int> q;
    for (int inputId : inputNodeIds) {
        q.push(inputId);
    }

    while (!q.empty()) {
        int v = q.front();
        q.pop();

        visitPredictNode(v);

        for (const auto& [u, conn] : adjacencyList[v]) {
            visitPredictNeighbor(conn);
            inDegrees[u]--;
            if (inDegrees[u] == 0) {
                q.push(u);
            }
        }
    }

    vector<double> output;
    for (int i = 0; i < outputNodeIds.size(); i++) {
        int dest = outputNodeIds.at(i);
        NodeInfo* outputNode = nodes.at(dest);
        output.push_back(outputNode->postActivationValue);
    }

    if (evaluating) {
        flush();
    } else {
        batchSize++;
        bool contribResult = contribute(instance.y, output.at(0));
    }

    return output;
}
// STUDENT TODO: IMPLEMENT
bool NeuralNetwork::contribute(double y, double p) {
    contributions.clear();
    for (int outputId : outputNodeIds) {
        contribute(outputId, y, p);  
    }

    for (int inputId : inputNodeIds) {
        // For input nodes, do not recursively call contribute() on them.
        // Instead, set their outgoingContribution to their postActivationValue.
        contributions[inputId] = nodes[inputId]->postActivationValue;
        for (auto& [neighborId, conn] : adjacencyList[inputId]) {
            double incomingContribution = contributions.count(neighborId) ? contributions[neighborId] : contribute(neighborId, y, p);
            visitContributeNeighbor(conn, incomingContribution, contributions[inputId]);
        }
    }

    return true;
}
// STUDENT TODO: IMPLEMENT
double NeuralNetwork::contribute(int nodeId, const double& y, const double& p) {
    if (std::find(inputNodeIds.begin(), inputNodeIds.end(), nodeId) != inputNodeIds.end()) {
        double val = nodes.at(nodeId)->postActivationValue;
        contributions[nodeId] = val;
        return val;
    }

    if (contributions.count(nodeId)) {
        return contributions[nodeId];
    }

    NodeInfo* currNode = nodes.at(nodeId);
    double outgoingContribution = 0;

    if (std::find(outputNodeIds.begin(), outputNodeIds.end(), nodeId) != outputNodeIds.end()) {
        outgoingContribution = p - y;
        visitContributeNode(nodeId, outgoingContribution);
        contributions[nodeId] = outgoingContribution;
        return outgoingContribution;
    }

    for (int srcId = 0; srcId < adjacencyList.size(); ++srcId) {
        if (adjacencyList[srcId].count(nodeId)) {
            Connection& conn = adjacencyList[srcId][nodeId];
            double srcContribution = contribute(srcId, y, p);
            visitContributeNeighbor(conn, srcContribution, outgoingContribution);
        }
    }

    if (std::find(inputNodeIds.begin(), inputNodeIds.end(), nodeId) == inputNodeIds.end()) {
        visitContributeNode(nodeId, outgoingContribution);
    }
    contributions[nodeId] = outgoingContribution;

    return outgoingContribution;
}
// STUDENT TODO: IMPLEMENT
bool NeuralNetwork::update() {

    double effectiveLearningRate = learningRate;

    for (NodeInfo* node : nodes) {
        node->bias -= effectiveLearningRate *  node->delta;
        node->delta = 0;
    }

    for (auto& connectionMap : adjacencyList) {
        for (auto& [destId, conn] : connectionMap) {
            // cout << "[DEBUG] Connection from " << conn.source << " to " << conn.dest << ", weight delta * batchSize: " << conn.delta * batchSize << endl;
            if (batchSize > 0) {
                conn.weight -= (effectiveLearningRate / batchSize) * conn.delta;
            } else {
                conn.weight -= effectiveLearningRate * conn.delta;
            }
            conn.delta = 0;
        }
    }

    flush();

    return true;
}




// Feel free to explore the remaining code, but no need to implement past this point

// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------
// ----------- YOU DO NOT NEED TO TOUCH THE REMAINING CODE -----------------------------------------------------------------







// Constructors
NeuralNetwork::NeuralNetwork() : Graph(0) {
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(int size) : Graph(size) {
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

NeuralNetwork::NeuralNetwork(string filename) : Graph() {
    // open file
    ifstream fin(filename);

    // error check
    if (fin.fail()) {
        cerr << "Could not open " << filename << " for reading. " << endl;
        exit(1);
    }

    // load network
    loadNetwork(fin);
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;

    // close file
    fin.close();
}

NeuralNetwork::NeuralNetwork(istream& in) : Graph() {
    loadNetwork(in);
    learningRate = 0.1;
    evaluating = false;
    batchSize = 0;
}

void NeuralNetwork::loadNetwork(istream& in) {
    int numLayers(0), totalNodes(0), numNodes(0), weightModifications(0), biasModifications(0); string activationMethod = "identity";
    string junk;
    in >> numLayers; in >> totalNodes; getline(in, junk);
    if (numLayers <= 1) {
        cerr << "Neural Network must have at least 2 layers, but got " << numLayers << " layers" << endl;
        exit(1);
    }

    // resize network to accomodate expected nodes.
    resize(totalNodes);
    this->size = totalNodes;

    int currentNodeId(0);

    vector<int> previousLayer;
    vector<int> currentLayer;
    for (int i = 0; i < numLayers; i++) {
        currentLayer.clear();
        //  For each layer

        // get nodes for this layer and activation method
        in >> numNodes; in >> activationMethod; getline(in, junk);

        for (int j = 0; j < numNodes; j++) {
            // For every node, add a new node to the network with proper activationMethod
            // initialize bias to 0.
            updateNode(currentNodeId, NodeInfo(activationMethod, 0, 0));
            // This node has an id of currentNodeId
            currentLayer.push_back(currentNodeId++);
        }

        if (i != 0) {
            // There exists a previous layer, now we set out connections
            for (int k = 0; k < previousLayer.size(); k++) {
                for (int w = 0; w < currentLayer.size(); w++) {

                    // Initialize an initial weight of a sample from the standard normal distribution
                    updateConnection(previousLayer.at(k), currentLayer.at(w), sample());
                }
            }
        }

        // Crawl forward.
        previousLayer = currentLayer;
        layers.push_back(currentLayer);
    }
    in >> weightModifications; getline(in, junk);
    int v(0),u(0); double w(0), b(0);

    // load weights by updating connections
    for (int i = 0; i < weightModifications; i++) {
        in >> v; in >> u; in >> w; getline(in , junk);
        updateConnection(v, u, w);
    }

    in >> biasModifications; getline(in , junk);

    // load biases by updating node info
    for (int i = 0; i < biasModifications; i++) {
        in >> v; in >> b; getline(in, junk);
        NodeInfo* thisNode = getNode(v);
        thisNode->bias = b;
    }

    setInputNodeIds(layers.at(0));
    setOutputNodeIds(layers.at(layers.size()-1));
}

void NeuralNetwork::visitPredictNode(int vId) {
    // accumulate bias, and activate
    NodeInfo* v = nodes.at(vId);
    v->preActivationValue += v->bias;
    v->activate();
}

void NeuralNetwork::visitPredictNeighbor(Connection c) {
    NodeInfo* v = nodes.at(c.source);
    NodeInfo* u = nodes.at(c.dest);
    u->preActivationValue += v->postActivationValue * c.weight;
}

void NeuralNetwork::visitContributeNode(int vId, double& outgoingContribution) {
    NodeInfo* v = nodes.at(vId);
    outgoingContribution *= v->derive();
    v->delta += outgoingContribution;
}

void NeuralNetwork::visitContributeNeighbor(Connection& c, double& incomingContribution, double& outgoingContribution) {
    NodeInfo* v = nodes.at(c.source);


    outgoingContribution += c.weight * incomingContribution;
    c.delta += incomingContribution * v->postActivationValue;

}
void NeuralNetwork::flush() {
    for (int i = 0; i < nodes.size(); i++) {
        nodes.at(i)->postActivationValue = 0;
        nodes.at(i)->preActivationValue = 0;
    }
    contributions.clear();
    batchSize = 0;
}

double NeuralNetwork::assess(string filename) {
    DataLoader dl(filename);
    return assess(dl);
}

double NeuralNetwork::assess(DataLoader dl) {
    bool stateBefore = evaluating;
    evaluating = true;
    double count(0);
    double correct(0);
    vector<double> output;
    for (int i = 0; i < dl.getData().size(); i++) {
        DataInstance di = dl.getData().at(i);
        output = predict(di);
        if (static_cast<int>(round(output.at(0))) == di.y) {
            correct++;
        }
        count++;
    }

    if (dl.getData().empty()) {
        cerr << "Cannot assess accuracy on an empty dataset" << endl;
        exit(1);
    }
    evaluating = stateBefore;
    return correct / count;
}


void NeuralNetwork::saveModel(string filename) {
    ofstream fout(filename);
    
    fout << layers.size() << " " << getNodes().size() << endl;
    for (int i = 0; i < layers.size(); i++) {
        NodeInfo* layerNode = getNodes().at(layers.at(i).at(0));
        string activationType = getActivationIdentifier(layerNode->activationFunction);

        fout << layers.at(i).size() << " " << activationType << endl;
    }

    int numWeights = 0;
    int numBias = 0;
    stringstream weightStream;
    stringstream biasStream;
    for (int i = 0; i < nodes.size(); i++) {
        numBias++;
        biasStream << i << " " << nodes.at(i)->bias << endl;

        for (auto j = adjacencyList.at(i).begin(); j != adjacencyList.at(i).end(); j++) {
            numWeights++;
            weightStream << j->second.source << " " << j->second.dest << " " << j->second.weight << endl;
        }
    }

    fout << numWeights << endl;
    fout << weightStream.str();
    fout << numBias << endl;
    fout << biasStream.str();

    fout.close();


}

ostream& operator<<(ostream& out, const NeuralNetwork& nn) {
    for (int i = 0; i < nn.layers.size(); i++) {
        out << "layer " << i << ": ";
        for (int j = 0; j < nn.layers.at(i).size(); j++) {
            out << nn.layers.at(i).at(j) << " ";
        }
        out << endl;
    }
    // outputs the nn in dot format
    out << static_cast<const Graph&>(nn) << endl;
    return out;
}
