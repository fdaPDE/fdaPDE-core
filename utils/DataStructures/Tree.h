#ifndef __TREE_H__
#define __TREE_H__

#include <memory>
#include <type_traits>
#include <unordered_map>
#include <queue>
#include <stack>
#include <functional>

namespace fdaPDE{
namespace core{
  
  // An implementation of a binary tree data structure

  // forward declaration
  template <typename T> class Node;
  template <typename T> class Tree;
  // define a node_ptr as a shared pointer to a node
  template <typename T> using node_ptr = std::shared_ptr<Node<T>>;

  enum LinkDirection {LEFT, RIGHT};

  // a tree node holding a general type T
  template <typename T> class Node {
  private:  
    T data_;                               // the actual stored data
    std::array<node_ptr<T>, 2> children_;  // children_[0] is the left child. children_[1] is the right child
    unsigned int key_;                     // unique node key
    node_ptr<T> father_;                   // pointer to father
    bool visited_ = false;                 // in a visiting algorithm this can be used to check if a node has been visited or not
    
  public:
    // constructor
    Node(T data, unsigned int key, const node_ptr<T>& father) : data_(data), key_(key), father_(father) {} ;
    
    // add child at first available position. Returns nullptr if there is no room
    node_ptr<T> addChild(const T& data, unsigned int key, const node_ptr<T>& father);
    // add child at node at the specified position. Returns nullptr if the index is already in use
    node_ptr<T> addChild(const T& data, unsigned int key, const node_ptr<T>& father, LinkDirection index);

    // a method to check if this node has no children (leaf node)
    bool isLeaf() const;

    // getters
    T getData()                              const { return data_;     }
    std::array<node_ptr<T>, 2> getChildren() const { return children_; }
    unsigned int getKey()                    const { return key_;      }
    node_ptr<T> getFather()                  const { return father_;   }

    // a node is visited if both its children are visited. 
    bool isVisited() const { return isLeaf() ? visited_ : children_[0]->visited_ && children_[1]->visited_; }

    // declare tree as friend
    friend Tree<T>;

    void setAsVisited()    { visited_ = true;  }
    void clearVisited()    { visited_ = false; }
  };

  // a binary tree storing objects of type T.
  template <typename T>
  class Tree {
  private:
    // pointer to root (from the root you have access to any other node)
    node_ptr<T> root_;
    // counter to keep track of how many nodes are stored in the tree. It also provides
    // a way for unique indexing nodes by ID
    unsigned int numNodes = 0;

    // auxiliary data structure mapping node indexes to pointers to nodes
    std::unordered_map<unsigned int, node_ptr<T> > nodeTable;
  
  public:
    // an empty tree
    Tree() = default;

    // initialize the tree from an already existing node_ptr. Usefull to build views
    Tree(const node_ptr<T>& root_ptr) : root_(root_ptr) { nodeTable[root_ptr->getKey()] = root_; }

    // initialize the tree with a non 0 ID for the root node
    Tree(const T& root, unsigned int rootID) {
      node_ptr<T> rootNode = std::make_shared<Node<T>>(root, rootID, nullptr);
      root_ = rootNode;

      // store reference to root in nodeTable
      nodeTable[rootID] = rootNode;
      numNodes++;
    }

    Tree(const T& root) : Tree(root, 0) {}
    
    // insert node at first available position, returns the pointer to the inserted node
    node_ptr<T> insert(const T& data);
    // insert node at first available position using the given ID as node identifier
    node_ptr<T> insert(const T& data, unsigned int ID);
    // insert node as child given the ID of the father. Do nothing and return nullptr if there is already a child in that direction.
    // This routine is usefull if the tree has to be built progressively using some external criterion
    node_ptr<T> insert(const T& data, unsigned int fatherID, LinkDirection direction);
    // the following overloading allow to specify the ID of the inserted node
    node_ptr<T> insert(const T& data, unsigned int ID, unsigned int fatherID, LinkDirection direction);
    
    // perform depth-first-search over this tree applying lambda function f at each node.
    template <typename F>
    typename std::enable_if<std::is_invocable<F, T>::value, void>::type
    DFS(const F& f) const;
    // general depth-first-search strategy accepting a type F. F must expose the following 3 methods:
    //  - leafAction(const node_ptr<T>&):       action to perform when a leaf is visited (leafs are visited ones).
    //  - firstVisitAction(const node_ptr<T>&): action to perform when a non-leaf node is visited for the first time
    //  - lastVisitAction(const node_ptr<T>&):  action to perform when a non-leaf node has been completely visited. This action is executed
    //                                          right before moving to father node
    // moreover F can have a state to store and perform additional computations during the visit
    template <typename F>
    typename std::enable_if<!std::is_invocable<F, T>::value, void>::type
    DFS(F& f) const;
    
    // getters
    node_ptr<T> getNode(unsigned int ID) const { return nodeTable.at(ID); }
    node_ptr<T> getRoot()                const { return root_;            }
    unsigned int getNumberOfNodes()      const { return numNodes;         }
  };

#include "Tree.tpp"
}}
  
#endif // __TREE_H__
