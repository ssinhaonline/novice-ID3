ID3 Implementation
Platform: Python 2.7
Developers: Hsien Ming Lee (hlee99) and Souradeep Sinha (ssinha04)
Linux Command Line:

python myID3.py [training dataset filename] [testing dataset filename] [(im)purity measure] [threshhold value]
Example: python myID3.py Sample_Trainer.csv Sample_Tester.csv gini 0.6

1. Name of the data sets: SampleTrainer.csv SampleTester.csv
2. Equal Width Binning was used to discretize the numerical attributes only.
3. Threshhold values tried against: 0.0, 0.1, 0.2, 0.65, 0.8
4. Successfully implemented Part 1, 2, 3 and 4.
5. Print format is not as was desired, as Python does not have a library for trees and without such a library it would take more time to code up a depth first search traversal. The format that we used can be visualized as an preorder. Each Parent is tagged as P and left and right children are tagged as L and P respectively. LP and RP means that a left/right child is a parent to another subtree.
6. The entire source code was coded from scratch. Design, pseudocode, mathematical functions and conceptual credits to Hsien Ming Lee. Encoding the buildTree function credits to Souradeep Sinha.
