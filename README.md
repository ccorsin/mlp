# mlp
Multilayer perception

# Objectif
Mettre en place un multilayerperceptron pour déceler si un cancer est malin ou bénin sur un dataset de diagnostics de cancer du sein réalisé au Wisconsin.
 
Développement d'un réseau de typefeedforward, c’est-à-dire que leflot d’informations circule de la couche d’entrée (input layer) vers la couche de sortie(output layer) défini par la présence d’une ou plusieurs couches cachées (hiddenlayers) ainsi qu’une interconnexion de tous les neurones d’une couche à la suivante.

# Dataset
Le dataset est fourni en ressources. Il s’agit d’un fichier csv de 32 colonnes, la colonne diagnosisest le diagnostic associé à toutes les features de la ligne actuelle,elle peut avoir comme valeur M ou B (pour malin ou bénin).

# Programmes 
Le programme d’entrainement utilise la backpropagation et la descente degradient pour apprendre sur les données d’entrainement et sauvegarde le modèle (la topologie et les poids du réseau) à la fin de son exécution.
