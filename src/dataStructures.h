#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>
#include <stdexcept>

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

template <class T> class DataBuffer;

class Iterator;

class Node
{
    friend class Iterator;
    template <class T> friend class DataBuffer;

protected:
    Node *next_, *prev_;

    // Set a node that works as both head and tail, this node doesn't hold any data
    // [h/t] next is the first data node
    // [h/t] prev is the last data node
    // Thus, forming a circular array
    // push_back() is only called in [h/t] node instance.
    void push_back(Node* node)
    {
        node->next_ = this; // new node->next is our head/tail node.
        node->prev_ = prev_; // new node->prev is previous last node.
        prev_->next_ = node; // previous last node->next is the new node.
        prev_ = node; // head/tail node->prev is the new node.
    }

    void unlink()
    {
        Node *next = next_, *prev = prev_;
        next->prev_ = prev;
        prev->next_ = next;
        next_ = this;
        prev_ = this;
    }

public:
    Node() :next_(this), prev_(this) {}
    ~Node() { unlink(); }
};

class Iterator
{
protected:
    Node* node_;
    Iterator(Node* node): node_(node) {}

public:

    // Prefix ++obj
    Iterator &operator++()
    {
        node_ = node_->next_;
        return *this;
    }

    // Postfix obj++
    Iterator operator++(int)
    {
        Iterator it(node_);
        ++(*this);
        return  it;
    }

    Iterator &operator--()
    {
        node_ = node_->prev_;
        return *this;
    }

    Iterator operator--(int)
    {
        Iterator it(node_);
        --(*this);
        return  it;
    }

    bool operator==(Iterator it) const { return node_ == it.node_; }
    bool operator!=(Iterator it) const { return !(*this == it); }
};

template <class T>
class DataBuffer
{
    // Node class that contains template data
    class NodeT : public Node
    {
        friend class DataBuffer<T>;
        T data_;
        NodeT(T data): data_(data) {}
    };

    // Iterator for NodeT
    template <class U>
    class IteratorT : public Iterator
    {
        friend class DataBuffer<T>;
        NodeT* node() const { return static_cast<NodeT *>(node_);}

    public:
        IteratorT(Node* node) : Iterator(node) {}
        U& operator*() { return node()->data_;}
        U* operator->() { return &node()->data_;}

        IteratorT operator- (int i)
        {
            IteratorT it(node_);
            while (i > 0)
            {
                it.node_ = it.node_->prev_;
                i--;
            }
            return it;
        }

        IteratorT operator+ (int i)
        {
            IteratorT it(node_);
            while (i > 0)
            {
                it.node_ = it.node_->next_;
                i--;
            }
            return it;
        }

        //iterator to const_iterator conversion
        operator IteratorT<U const>() const { return node_; }
    };

    // DataBuffer member attributes
    int capacity_;
    int size_;
    Node list_; // This is our head/tail node pointer.

public:
    using iterator = IteratorT<T>;
    using const_iterator = IteratorT<T const>;

    DataBuffer(): capacity_(5), size_(0), list_(Node()) {}
    DataBuffer(int capacity): capacity_(capacity), size_(0), list_(Node()) {}
    ~DataBuffer() { clear(); }

    bool empty() const { return list_.next_ == &list_; };
    int size() const { return size_; }

    void SetCapacity(int capacity) { capacity_ = capacity; }

    iterator begin() { return list_.next_; }
    iterator end() { return &list_; }

    void push_back(T data)
    {
        if (size_ < capacity_)
        {
            list_.push_back(new NodeT(data));
            size_++;
        }
        else
        {
            erase(begin());
            list_.push_back(new NodeT(data));
        }
    }

    void erase(const_iterator it)
    {
        delete it.node();
        --size_;
    }

    void clear()
    {
        while (!empty())
            erase(begin());
    }
};


#endif /* dataStructures_h */
