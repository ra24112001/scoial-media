from msilib.schema import RadioButton
import sys
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
import community 
from networkx.algorithms.community import girvan_newman

from math import cos, sin
from PyQt5.QtWidgets import QHBoxLayout, QMessageBox, QLabel,QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit,QFileDialog,QCheckBox,QComboBox
from PyQt5.QtCore import QSize
from networkx.algorithms.cuts import conductance
from networkx.algorithms.community import greedy_modularity_communities, modularity
from sklearn.metrics.cluster import normalized_mutual_info_score
from PyQt5.QtWidgets import QRadioButton


class SocialNetworkAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Social Network Analyzer")
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        
     
        self.lbl_node_size = QLabel("Node Size:", self)
        self.layout.addWidget(self.lbl_node_size)
        self.txt_node_size = QLineEdit(self)
        self.layout.addWidget(self.txt_node_size)

        self.lbl_node_color = QLabel("Node Color (e.g., 'red', '#00FF00'):", self)
        self.layout.addWidget(self.lbl_node_color)
        self.txt_node_color = QLineEdit(self)
        self.layout.addWidget(self.txt_node_color)

        self.lbl_node_label = QLabel("Node Label:", self)
        self.layout.addWidget(self.lbl_node_label)
        self.txt_node_label = QLineEdit(self)
        self.layout.addWidget(self.txt_node_label)

        
        self.lbl_node_shape = QLabel("Node Shape (e.g., 'o', 's', '^',):", self)
        self.layout.addWidget(self.lbl_node_shape)
        self.txt_node_shape = QLineEdit(self)
        self.layout.addWidget(self.txt_node_shape)

        self.lbl_edge_color = QLabel("Edge Color (e.g., 'blue', '#FFA500'):", self)
        self.layout.addWidget(self.lbl_edge_color)
        self.txt_edge_color = QLineEdit(self)
        self.layout.addWidget(self.txt_edge_color)

        self.lbl_edge_label = QLabel("Edge Label:", self)
        self.layout.addWidget(self.lbl_edge_label)
        self.txt_edge_label = QLineEdit(self)
        self.layout.addWidget(self.txt_edge_label)


        self.layout.addWidget(QLabel("Select Layout:"))
        self.layout_combobox = QComboBox(self)
        self.layout_combobox.addItems(["None","Fruchterman-Reingold", "Radial Axis Layout"])
        self.layout.addWidget(self.layout_combobox)
       
        self.layout_combobox.setCurrentIndex(0)

        self.layout.addWidget(QLabel("Select Partitioning Attribute:"))
        self.partition_combobox = QComboBox(self)
        self.partition_combobox.addItems(["Class", "Gender"])
        self.layout.addWidget(self.partition_combobox)

        self.lbl_threshold = QLabel("Value to Calculate :", self)
        self.layout.addWidget(self.lbl_threshold)
        self.txt_threshold = QLineEdit(self)
        self.layout.addWidget(self.txt_threshold)

        button_size = QSize(200, 30)
        self.button_layout = QHBoxLayout()

        self.btn_apply_attributes = QPushButton("Apply Node and Edge Attributes", self)
        self.btn_apply_attributes.clicked.connect(self.apply_attributes)
        self.btn_apply_attributes.setFixedSize(button_size)
        self.button_layout.addWidget(self.btn_apply_attributes)

        self.btn_visualize_eigenvector = QPushButton("Visualize Eigenvector Centrality", self)
        self.btn_visualize_eigenvector.clicked.connect(self.visualize_eigenvector_centrality)
        self.btn_visualize_eigenvector.setFixedSize(button_size)
        self.button_layout.addWidget(self.btn_visualize_eigenvector)
        
        self.btn_visualize_degree = QPushButton("Visualize Degree Centrality", self)
        self.btn_visualize_degree.clicked.connect(self.visualize_degree_centrality)
        self.btn_visualize_degree.setFixedSize(button_size)
        self.button_layout.addWidget(self.btn_visualize_degree)

        self.btn_visualize_closeness = QPushButton("Visualize Closeness Centrality", self)
        self.btn_visualize_closeness.clicked.connect(self.visualize_closeness_centrality)
        self.btn_visualize_closeness.setFixedSize(button_size)
        self.button_layout.addWidget(self.btn_visualize_closeness)

        self.btn_visualize_betweenness = QPushButton("Visualize Betweenness Centrality", self)
        self.btn_visualize_betweenness.clicked.connect(self.visualize_betweenness_centrality)
        self.btn_visualize_betweenness.setFixedSize(button_size)
        self.button_layout.addWidget(self.btn_visualize_betweenness)

        
        

        self.btn_analyze_network = QPushButton("Analyze Network", self)
        self.btn_analyze_network.clicked.connect(self.analyze_network)
        self.btn_analyze_network.setFixedSize(button_size)
        self.button_layout.addWidget(self.btn_analyze_network)




        self.btn_partition_community = QPushButton("Partition Community", self)
        self.btn_partition_community.clicked.connect(self.partition_community)
        self.btn_partition_community.setFixedSize(button_size)
        self.button_layout.addWidget(self.btn_partition_community)



        # Button to compare community detection algorithms
        self.btn_compare_detection = QPushButton("Compare Community Detection", self)
        self.btn_compare_detection.clicked.connect(self.compare_community_detection)
        self.layout.addWidget(self.btn_compare_detection)

        self.layout.addLayout(self.button_layout)

        self.button1_layout = QHBoxLayout()

        self.btn_run_modularity = QPushButton("Run Modularity", self)
        self.btn_run_modularity.clicked.connect(self.run_modularity)
        self.btn_run_modularity.setFixedSize(button_size)
        self.button1_layout.addWidget(self.btn_run_modularity)

        # Button to run conductance calculation
        self.btn_run_conductance = QPushButton("Run Conductance", self)
        self.btn_run_conductance.clicked.connect(self.run_conductance)
        self.btn_run_conductance.setFixedSize(button_size)
        self.button1_layout.addWidget(self.btn_run_conductance)

        # Button to run NMI calculation
        self.btn_run_nmi = QPushButton("Run NMI", self)
        self.btn_run_nmi.setFixedSize(button_size)
        self.btn_run_nmi.clicked.connect(self.run_nmi)
        self.button1_layout.addWidget(self.btn_run_nmi)
        
        # Button to run Coverge calculation
        self.btn_run_cov = QPushButton("Run Coverage", self)
        self.btn_run_cov.setFixedSize(button_size)
        self.btn_run_cov.clicked.connect(self.coverage)
        self.button1_layout.addWidget(self.btn_run_cov)

        self.btn_visualize_pagerank = QPushButton("Visualize PageRank", self)
        self.btn_visualize_pagerank.clicked.connect(self.visualize_pagerank_scores)
        self.btn_visualize_pagerank.setFixedSize(button_size)
        self.button1_layout.addWidget(self.btn_visualize_pagerank)

        self.layout.addLayout(self.button1_layout)

        self.lbl_min_value = QLabel("Min Value:20", self)
        self.txt_min_value = QLineEdit(self)
        self.layout.addWidget(self.lbl_min_value)
        self.layout.addWidget(self.txt_min_value)
        
        self.lbl_max_value = QLabel("Max Value:134", self)
        self.txt_max_value = QLineEdit(self)
        self.layout.addWidget(self.lbl_max_value) 
        self.layout.addWidget(self.txt_max_value)      
         # Button to apply degree filter      
        self.btn_apply_degree_filter = QPushButton("Apply Degree Filter", self)
        self.btn_apply_degree_filter.clicked.connect(self.apply_degree_filter)
        self.layout.addWidget(self.btn_apply_degree_filter)

        self.txt_output = QTextEdit(self)
        self.layout.addWidget(self.txt_output)
        self.radio_directed = QRadioButton("Directed", self)
        self.radio_undirected = QRadioButton("Undirected", self)
        self.layout.addWidget(self.radio_directed)
        self.layout.addWidget(self.radio_undirected)

        self.radio_undirected.setChecked(True)

        self.radio_directed.toggled.connect(self.run_code)
        self.radio_undirected.toggled.connect(self.run_code)

        self.setCentralWidget(self.central_widget)

        # Create an empty graph
        self.G = nx.Graph()
    def run_code(self):
        # This function will be triggered whenever the radio buttons are toggled
        if self.radio_undirected.isChecked():
            print("Undirected graph selected")
            # Add your logic here to handle undirected graph
            if self.G is not None:
                self.G = self.G.to_undirected()
            else:
                print("No graph loaded.")
        elif self.radio_directed.isChecked():
            print("Directed graph selected")
            # Add your logic here to handle directed graph
            if self.G is not None:
                self.G = self.G.to_directed()
            else:
                print("No graph loaded.") 

   
         
    def apply_attributes(self):
        if hasattr(self, 'G'):
            # Get user-defined attributes
            node_size = int(self.txt_node_size.text()) if self.txt_node_size.text() else 50
            node_color = self.txt_node_color.text() if self.txt_node_color.text() else 'skyblue'
            node_shape = self.txt_node_shape.text() if self.txt_node_shape.text() else 'o'
            node_label = self.txt_node_label.text() if self.txt_node_label.text() else ''
            edge_color = self.txt_edge_color.text() if self.txt_edge_color.text() else 'black'
            edge_label = self.txt_edge_label.text() if self.txt_edge_label.text() else ''

            
            # Apply attributes to nodes and edges
            nx.set_node_attributes(self.G, {node: {'size': node_size, 'color': node_color, 'shape': node_shape, 'label': node_label} for node in self.G.nodes()})
            nx.set_edge_attributes(self.G, {edge: {'color': edge_color, 'label': edge_label} for edge in self.G.edges()})

            # Visualize the graph
            layout_type = self.layout_combobox.currentText()

            if layout_type == "Fruchterman-Reingold":
                pos = nx.spring_layout(self.G)
            elif layout_type == "Radial Axis Layout":
                pos = self.radial_layout(self.G)
            elif layout_type == "None":
                pos = None

            if pos is not None:
                node_sizes = [self.G.nodes[node]['size'] for node in self.G.nodes()]
                node_colors = [self.G.nodes[node]['color'] for node in self.G.nodes()]
                edge_colors = [self.G.edges[edge]['color'] for edge in self.G.edges()]
                
                nx.draw(self.G, pos=pos, node_size=node_sizes, node_color=node_colors, node_shape=node_shape, edge_color=edge_colors, with_labels=True, labels=nx.get_node_attributes(self.G, 'label'))
                plt.title(f"{layout_type} Layout")
                plt.show()
            else:
                # Visualize the graph with a default layout
                nx.draw(self.G, with_labels=True, node_size=node_size, node_color=node_color, node_shape=node_shape, edge_color=edge_color,labels=nx.get_node_attributes(self.G, 'label'))
                plt.title("Default Layout")
                plt.show()

                self.txt_output.clear()
                self.txt_output.append("Node and Edge Attributes Applied Successfully!\n")    

        else:
            self.txt_output.append("Please load a network first!\n")
        if hasattr(self, 'G'):
            # Get user-defined attributes
            node_size = int(self.txt_node_size.text()) if self.txt_node_size.text() else 50
            node_color = self.txt_node_color.text() if self.txt_node_color.text() else 'skyblue'
            node_shape = self.txt_node_shape.text() if self.txt_node_shape.text() else 'o'
            node_label = self.txt_node_label.text() if self.txt_node_label.text() else ''
            edge_color = self.txt_edge_color.text() if self.txt_edge_color.text() else 'black'
            edge_label = self.txt_edge_label.text() if self.txt_edge_label.text() else ''

            if node_label is None:
                node_label = {node: str(node) for node in self.G.nodes()}  # Use node IDs as labels

            # Apply attributes to nodes and edges
            nx.set_node_attributes(self.G, {node: {'size': node_size, 'color': node_color, 'shape': node_shape, 'label': node_label} for node in self.G.nodes()})
            nx.set_edge_attributes(self.G, {edge: {'color': edge_color, 'label': edge_label} for edge in self.G.edges()})

            # Visualize the graph
            layout_type = self.layout_combobox.currentText()

            if layout_type == "Fruchterman-Reingold":
                pos = nx.spring_layout(self.G)
            elif layout_type == "Radial Axis Layout":
                pos = self.radial_layout(self.G)
            elif layout_type == "None":
                pos = None

            if pos is not None:
                node_sizes = [self.G.nodes[node]['size'] for node in self.G.nodes()]
                node_colors = [self.G.nodes[node]['color'] for node in self.G.nodes()]
                edge_colors = [self.G.edges[edge]['color'] for edge in self.G.edges()]
                
                nx.draw(self.G, pos=pos, node_size=node_sizes, node_color=node_colors, node_shape=node_shape, edge_color=edge_colors, with_labels=True, labels=nx.get_node_attributes(self.G, 'label'))
                plt.title(f"{layout_type} Layout")
                plt.show()
            else:
                # Visualize the graph with a default layout
                nx.draw(self.G, with_labels=True, node_size=node_size, node_color=node_color, node_shape=node_shape, edge_color=edge_color,labels=nx.get_node_attributes(self.G, 'label'))
                plt.title("Default Layout")
                plt.show()

                self.txt_output.clear()
                self.txt_output.append("Node and Edge Attributes Applied Successfully!\n")    

        else:
            self.txt_output.append("Please load a network first!\n")

    def visualize_eigenvector_centrality(self):
        if hasattr(self, 'G'):
            centrality_values = nx.eigenvector_centrality(self.G)
            self.visualize_centrality(centrality_values, "Eigenvector Centrality")
        else:
            self.txt_output.append("Please load a network first!\n")
 
    def visualize_degree_centrality(self):
        if hasattr(self, 'G'):
            centrality_values = nx.degree_centrality(self.G)
            self.visualize_centrality(centrality_values, "Degree Centrality")
        else:
            self.txt_output.append("Please load a network first!\n")

    # Function to visualize closeness centrality
    def visualize_closeness_centrality(self):
        if hasattr(self, 'G'):
            centrality_values = nx.closeness_centrality(self.G)
            self.visualize_centrality(centrality_values, "Closeness Centrality")
        else:
            self.txt_output.append("Please load a network first!\n")
    # Function to visualize betweenness centrality
    def visualize_betweenness_centrality(self):
        if hasattr(self, 'G'):
            centrality_values = nx.betweenness_centrality(self.G)
            self.visualize_centrality(centrality_values, "Betweenness Centrality")
        else:
            self.txt_output.append("Please load a network first!\n")

    # General function to visualize centrality
    def visualize_centrality(self, centrality_values, centrality_name):
        threshold = float(self.txt_threshold.text()) if self.txt_threshold.text() else 0.5
        node_size = int(self.txt_node_size.text()) if self.txt_node_size.text() else 20
        filtered_nodes = [node for node, centrality in centrality_values.items() if centrality >= threshold]

        layout_type = self.layout_combobox.currentText() 
        plt.figure()
        pos = nx.spring_layout(self.G)  # Use spring layout for visualization
        nx.draw_networkx_nodes(self.G, pos=pos, nodelist=filtered_nodes, node_color='red', node_size=node_size)
        #nx.draw_networkx_edges(self.G, pos=pos, alpha=0.5)
        plt.title(f"{layout_type} Layout - {centrality_name} >= {threshold}")
        plt.axis('on')
        plt.axis('equal')

        plt.show()
        
    def apply_degree_filter(self):
        if hasattr(self, 'G'):
            # Retrieve degree range from slider
            degree_range = self.slider_degree_range.value()

            # Filter nodes based on the degree range
            filtered_nodes = [node for node, degree in self.G.degree() if degree >= degree_range]

            # Visualize the filtered nodes
            layout_type = self.layout_combobox.currentText() 
            plt.figure()
            nx.draw(self.G, pos=nx.spring_layout(self.G), with_labels=True, node_color='skyblue', node_size=300, labels=nx.get_node_attributes(self.G, 'label'))
            nx.draw_networkx_nodes(self.G, pos=nx.spring_layout(self.G), nodelist=filtered_nodes, node_color='red', node_size=300)
            plt.title(f"{layout_type} Layout - Degree Range: >= {degree_range}")
            plt.show()    
    
    def analyze_network(self):
        if hasattr(self, 'G'):
            # Calculate basic network metrics
            degree_distribution = nx.degree_histogram(self.G)
            total_nodes = len(self.G.nodes())
            degree_probabilities = [count / total_nodes for count in degree_distribution]
            clustering_coefficient = nx.average_clustering(self.G)
            average_path_length = nx.average_shortest_path_length(self.G)

            # Display network metrics
            self.txt_output.clear()
            self.txt_output.append("Degree Distribution (Probabilities):\n")
            for degree, prob in enumerate(degree_probabilities):
                self.txt_output.append(f"Degree {degree}: {prob:.4f}")
            self.txt_output.append(f"\nClustering Coefficient: {clustering_coefficient}\n")
            self.txt_output.append(f"Average Path Length: {average_path_length}\n")
            self.txt_output.append("-------------------------------------------- ")


            # Centrality measures
            centrality_measures = {
                "Degree Centrality": nx.degree_centrality(self.G),
                "Closeness Centrality": nx.closeness_centrality(self.G),
                "Betweenness Centrality": nx.betweenness_centrality(self.G),
                "Eigenvector Centrality": nx.eigenvector_centrality(self.G)
            }
            self.txt_output.append("Centrality Measures:\n")
            for measure, values in centrality_measures.items():
                self.txt_output.append(f"{measure}:\n")
                for node, centrality in values.items():
                    self.txt_output.append(f"{node}: {centrality:.15f}")
                self.txt_output.append("\n")
                self.txt_output.append("-------------------------------------------- ")

            pagerank_scores = nx.pagerank(self.G)
            self.txt_output.append("PageRank Scores:\n")
            for node, score in pagerank_scores.items():
                self.txt_output.append(f"{node}: {score:.15f}")    

        else:
            self.txt_output.append("Please load a network first!\n")
        
    def radial_layout(self, G):
        """Compute positions for the nodes of a graph in a radial layout."""
        diameter = nx.diameter(G)
        radius = diameter / 2.0
        center = (0, 0)
        theta = 2.0 * 3.141592653589793238462643383279502884197169399375 / len(G)
        pos = {}
        for i, node in enumerate(G.nodes()):
            angle = i * theta
            x = center[0] + radius * cos(angle)
            y = center[1] + radius * sin(angle)
            pos[node] = (x, y)
        return pos        
    def update_degree_range_label(self, value):
        self.lbl_degree_range_value.setText(str(value))

    def update_min_max_labels(self, value):
        min_value = self.slider_degree_range.minimum()
        max_value = self.slider_degree_range.maximum()
        self.lbl_min_max_values.setText(f"({min_value} - {max_value})")

    def apply_degree_filter(self):
        if hasattr(self, 'G'):
            # Retrieve degree range from user input
            min_text = self.txt_min_value.text()
            max_text = self.txt_max_value.text()

            # Set default values if fields are empty
            min_value = int(min_text) if min_text else 20
            max_value = int(max_text) if max_text else 134

            # Filter nodes based on the degree range
            filtered_nodes = [node for node, degree in self.G.degree() if min_value <= degree <= max_value]

            # Create a subgraph with only the filtered nodes
            filtered_graph = self.G.subgraph(filtered_nodes)

            # Visualize the filtered nodes
            layout_type = self.layout_combobox.currentText()
            plt.figure()
            nx.draw(filtered_graph, pos=nx.spring_layout(filtered_graph), with_labels=True, node_color='skyblue', node_size=300, labels=nx.get_node_attributes(filtered_graph, 'label'))
            plt.title(f"{layout_type} Layout - Degree Range: {min_value} to {max_value}")
            plt.show()
        else:
            self.txt_output.append("Please load a network first!\n")

    def compare_community_detection(self):
        if hasattr(self, 'G'):
            # Run Girvan-Newman algorithm
            girvan_communities_iter = girvan_newman(self.G)
            girvan_communities = tuple(sorted(c) for c in next(girvan_communities_iter))
            girvan_num_communities = len(girvan_communities)
            girvan_mod_score = modularity(self.G, girvan_communities)
            
            # Run Louvain algorithm
            partition = community.best_partition(self.G)
            louvain_communities = {}
            for node, community_id in partition.items():
                if community_id not in louvain_communities:
                    louvain_communities[community_id] = set()
                louvain_communities[community_id].add(node)
            louvain_num_communities = len(louvain_communities)
            louvain_mod_score = modularity(self.G, list(louvain_communities.values()))
            
            # Display results in QMessageBox
            result_text = f"Girvan-Newman Algorithm:\nNumber of Communities: {girvan_num_communities}\nModularity Score: {girvan_mod_score}\n\n" \
                          f"Louvain Algorithm:\nNumber of Communities: {louvain_num_communities}\nModularity Score: {louvain_mod_score}"
            QMessageBox.information(self, "Community Detection Comparison Result", result_text)
        else:
            self.txt_output.append("Please load a network first!\n")


   

    def calculate_modularity(self, communities):
        
    
        if hasattr(self, 'G'):
            # Perform community detection using Girvan-Newman algorithm
            girvan_newman_communities = self.detect_girvan_newman_communities()
            
            # Convert the communities to a human-readable format
            community_str = ""
            for idx, community_set in enumerate(girvan_newman_communities, start=1):
                community_str += f"Community {idx}: {community_set}\n"
            
            # Show the results in a pop-up window
            QMessageBox.information(self, "Girvan-Newman Community Detection", community_str)
        else:
            self.txt_output.append("Please load a network first!\n")
   
    def load_network_from_csv(self):
        # Prompt the user to select CSV files
        nodes_file, _ = QFileDialog.getOpenFileName(self, 'Open Nodes CSV File', '', 'CSV Files (*.csv)')
        edges_file, _ = QFileDialog.getOpenFileName(self, 'Open Edges CSV File', '', 'CSV Files (*.csv)')
        
        if nodes_file and edges_file:  # Ensure files were selected
            # Read nodes and edges data from CSV files
            nodes_df = pd.read_csv(nodes_file)
            edges_df = pd.read_csv(edges_file)

            # Add nodes to the graph
            self.G.add_nodes_from(nodes_df['ID'])

            # Add edges to the graph
            self.G.add_edges_from(zip(edges_df['Source'], edges_df['Target']))

            num_nodes = len(self.G.nodes())
            num_edges = len(self.G.edges())

            if 'Class' in nodes_df.columns:
                nx.set_node_attributes(self.G, nodes_df.set_index('ID')['Class'].to_dict(), 'Class')
            if 'Gender' in nodes_df.columns:
                nx.set_node_attributes(self.G, nodes_df.set_index('ID')['Gender'].to_dict(), 'Gender')
            



            self.txt_output.append(f"Network loaded successfully!\nNumber of nodes: {num_nodes}\nNumber of edges: {num_edges}\n")
        else:
            self.txt_output.append("Please select both nodes and edges CSV files!\n")

    def partition_community(self):
        if hasattr(self, 'G'):
            # Get the selected attribute for partitioning
            selected_attribute = self.partition_combobox.currentText()

            # Apply community detection algorithm based on the selected attribute
            partition = self.detect_community_by_attribute(selected_attribute)

            # Visualize the partitioned communities
            self.visualize_partition(partition, title=f" {selected_attribute}")
            
        else:
            self.txt_output.append("Please load a network first!\n")


    def detect_community_by_attribute(self, attribute):
        # Group nodes based on the specified attribute
        attribute_values = nx.get_node_attributes(self.G, attribute)
        community_partition = {}
        for node, attr_value in attribute_values.items():
            if attr_value not in community_partition:
                community_partition[attr_value] = set()
            community_partition[attr_value].add(node)
        return community_partition

    def visualize_partition(self, partition, title):
        # Calculate number of communities
        num_communities = len(partition)
        num_rows = 1
        num_cols = num_communities

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))

        colors = plt.cm.tab10.colors
        num_colors_needed = num_communities
        while len(colors) < num_colors_needed:
            colors += colors

        # Visualize each partitioned community
        for idx, (attr_value, community_set) in enumerate(partition.items(), start=1):
            # Create a subgraph for the community
            community_graph = self.G.subgraph(community_set)

            # Determine subplot index
            if num_communities > 1:
                ax = axes[idx-1]
            else:
                ax = axes

            # Visualize the community
            pos = nx.spring_layout(community_graph)  # or use your preferred layout
            nx.draw(community_graph, pos=pos, with_labels=True, ax=ax, node_color=colors[idx], node_size=100)
            ax.set_title(f"{title}: {attr_value}")

        # Adjust layout
        plt.tight_layout()
        plt.show()
    def run_modularity(self):
        try:
            # Find communities using greedy modularity algorithm
            communities = list(greedy_modularity_communities(self.G))

            # Calculate modularity of communities
            modularity_value = modularity(self.G, communities)

            Communities = "Communities: {}".format(len(communities)) + '\n' + "Modularity value: {}".format(
                modularity_value)
            QMessageBox.information(self, "Modularity", Communities)
        except:
            QMessageBox.critical(self, "Error", "Invalid input: please enter a list of edges")

    def run_conductance(self):
        try:
            # Find the communities using the greedy modularity algorithm
            communities = greedy_modularity_communities(self.G)

            # Calculate the conductance for each community
            cond = ''
            for community in communities:
                if len(community) == 0:
                    # Skip empty communities
                    continue
                community_edges = self.G.subgraph(community).edges()
                complement_edges = self.G.subgraph(set(self.G.nodes()) - set(community)).edges()
                volume_community = sum(self.G[u][v].get('weight', 1) for u, v in community_edges)
                volume_complement = sum(self.G[u][v].get('weight', 1) for u, v in complement_edges)
                if volume_community == 0 or volume_complement == 0:
                    # Skip communities with no edges or complement with no edges
                    continue
                conductance_value = conductance(self.G, community)
                print(f"Community {community} has conductance {conductance_value}")
                cond += 'Community {}, has conductance {}'.format(community, 2 * conductance_value) + '\n'
            QMessageBox.information(self, "Conductance", cond)
        except:
            QMessageBox.critical(self, "Error", "Invalid input: please enter a list of edges")

    def run_nmi(self):
        try:
            partition1 = community.best_partition(self.G)
            partition2 = community.best_partition(self.G)

            # Convert the partitions into lists of cluster labels
            labels1 = [partition1[node] for node in self.G.nodes()]
            labels2 = [partition2[node] for node in self.G.nodes()]

            true_labels = [partition1.get(node) for node in self.G.nodes()]
            predicted_labels = [partition1[node] for node in self.G.nodes()]
            

            # Compute the NMI between the two clusterings
            nmi = normalized_mutual_info_score(labels1, labels2)
            QMessageBox.information(self, "NMI", "Normalized Mutual Information = {}".format(nmi))
        except:
            QMessageBox.critical(self, "Error", "Invalid input: please enter a list of edges")

    def coverage(self):
        try:
            communities = list(greedy_modularity_communities(self.G))

            # Calculate the total coverage
            total_coverage = 0
            for comm in communities:
                nodes_in_comm = set(comm)
                total_coverage += len(nodes_in_comm) / len(self.G.nodes)

            # Display the total coverage in one message box
            QMessageBox.information(self, "Coverage", f"Total coverage = {total_coverage}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


    def visualize_pagerank(self, pagerank_scores):
        threshold = float(self.txt_threshold.text()) if self.txt_threshold.text() else 0.5
        node_size = int(self.txt_node_size.text()) if self.txt_node_size.text() else 20

        # Filter nodes based on the threshold
        filtered_nodes = [node for node, score in pagerank_scores.items() if score >= threshold]

        layout_type = self.layout_combobox.currentText()
        plt.figure()
        pos = nx.spring_layout(self.G)  # Use spring layout for visualization
        nx.draw_networkx_nodes(self.G, pos=pos, nodelist=filtered_nodes, node_color='red', node_size=node_size)
        plt.title(f"{layout_type} Layout - PageRank >= {threshold}")
        plt.axis('on')
        plt.axis('equal')

        plt.show()

    def visualize_pagerank_scores(self):
        if hasattr(self, 'G'):
            try:
                # Calculate PageRank scores
                pagerank_scores = nx.pagerank(self.G)

                # Visualize PageRank scores
                self.visualize_pagerank(pagerank_scores)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
        else:
            self.txt_output.append("Please load a network first!\n")





if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SocialNetworkAnalyzer()
    window.load_network_from_csv()
    window.setGeometry(100, 100, 800, 600)
    
    window.show()
    sys.exit(app.exec_())

