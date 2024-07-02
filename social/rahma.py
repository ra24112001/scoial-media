import sys
import csv
import networkx as nx
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QColorDialog, QInputDialog, QLabel, QLineEdit


class NetworkVisualizationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.graph = nx.Graph()
        self.load_data()

        self.setWindowTitle("Network Visualization App")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Add buttons
        self.node_attr_button = QPushButton("Define Node Attributes")
        self.node_attr_button.clicked.connect(self.define_node_attributes)
        layout.addWidget(self.node_attr_button)

        self.edge_attr_button = QPushButton("Define Edge Attributes")
        self.edge_attr_button.clicked.connect(self.define_edge_attributes)
        layout.addWidget(self.edge_attr_button)

        # Add a widget to the main window
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def load_data(self):
        # Load node and edge data from CSV files
        with open('Node.csv', 'r') as node_file:
            node_reader = csv.reader(node_file)
            for row in node_reader:
                self.graph.add_node(row[0])

        with open('Edges.csv', 'r') as edge_file:
            edge_reader = csv.reader(edge_file)
            for row in edge_reader:
                self.graph.add_edge(row[0], row[1])

    def define_node_attributes(self):
        # Define custom node attributes interactively
        for node in self.graph.nodes():
            dialog = NodeAttributesDialog()
            if dialog.exec_():
                size = dialog.size_input.text()
                color = dialog.color_label.text()
                label = dialog.label_input.text()
                shape = dialog.shape_input.text()

                self.graph.nodes[node]['size'] = size
                self.graph.nodes[node]['color'] = color
                self.graph.nodes[node]['label'] = label
                self.graph.nodes[node]['shape'] = shape

        # Visualize the graph with node attributes
        self.visualize_graph()

    def define_edge_attributes(self):
        # Define custom edge attributes interactively
        for edge in self.graph.edges():
            dialog = EdgeAttributesDialog()
            if dialog.exec_():
                weight = dialog.weight_input.text()
                color = dialog.color_label.text()
                label = dialog.label_input.text()

                self.graph.edges[edge]['weight'] = weight
                self.graph.edges[edge]['color'] = color
                self.graph.edges[edge]['label'] = label

        # Visualize the graph with edge attributes
        self.visualize_graph()

    def visualize_graph(self):
        # Clear previous visualization
        plt.clf()

        # Visualize the graph
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=3000, node_color=[self.graph.nodes[node]['color'] for node in self.graph.nodes()],
                labels={node: self.graph.nodes[node]['label'] for node in self.graph.nodes()}, font_size=12, font_color='black',
                edge_color=[self.graph.edges[edge]['color'] for edge in self.graph.edges()], width=2, alpha=0.7,
                connectionstyle='arc3, rad = 0.1')

        plt.show()


class NodeAttributesDialog(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Define Node Attributes")
        self.layout = QVBoxLayout()

        self.size_label = QLabel("Size:")
        self.size_input = QLineEdit()
        self.layout.addWidget(self.size_label)
        self.layout.addWidget(self.size_input)

        self.color_button = QPushButton("Choose Color")
        self.color_button.clicked.connect(self.choose_color)
        self.layout.addWidget(self.color_button)
        self.color_label = QLabel()
        self.layout.addWidget(self.color_label)

        self.label_label = QLabel("Label:")
        self.label_input = QLineEdit()
        self.layout.addWidget(self.label_label)
        self.layout.addWidget(self.label_input)

        self.shape_label = QLabel("Shape:")
        self.shape_input = QLineEdit()
        self.layout.addWidget(self.shape_label)
        self.layout.addWidget(self.shape_input)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_label.setText(color.name())

    def accept(self):
        super().accept()


class EdgeAttributesDialog(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Define Edge Attributes")
        self.layout = QVBoxLayout()

        self.weight_label = QLabel("Weight:")
        self.weight_input = QLineEdit()
        self.layout.addWidget(self.weight_label)
        self.layout.addWidget(self.weight_input)

        self.color_button = QPushButton("Choose Color")
        self.color_button.clicked.connect(self.choose_color)
        self.layout.addWidget(self.color_button)
        self.color_label = QLabel()
        self.layout.addWidget(self.color_label)

        self.label_label = QLabel("Label:")
        self.label_input = QLineEdit()
        self.layout.addWidget(self.label_label)
        self.layout.addWidget(self.label_input)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_label.setText(color.name())

    def accept(self):
        super().accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NetworkVisualizationApp()
    window.show()
    sys.exit(app.exec_())
