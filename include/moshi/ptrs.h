#pragma once

#include <vector>

// just automatically deletes pointers on destruction
template<typename T>
class own_ptr {
public:
    T * ptr;
    own_ptr() {
        ptr = NULL;
    }
    own_ptr( T * ptr ) {
        this->ptr = ptr;
    }
    void reset() {
        if ( ! ptr )
            return;
        delete ptr;
        ptr = NULL;
    }
    ~own_ptr() {
        reset();
    }
    own_ptr<T> & operator=(T * ptr) {
        if ( ptr == this->ptr )
            return *this;
        reset();
        this->ptr = ptr;
        return *this;
    }
    T & operator*() {
        return *ptr;
    }
    T * operator->() {
        return ptr;
    }
    operator T *() {
        return ptr;
    }
    operator const T *() const {
        return ptr;
    }
    own_ptr( const own_ptr<T>& ) = delete; 
    own_ptr<T> & operator=( own_ptr<T>& ) = delete;
};

// just a std::vector that deletes pointers on destruction
template<typename T>
class own_ptr_vector : public std::vector<T*> {
public:
    using std::vector<T*>::vector; // Inherit constructors
    ~own_ptr_vector() {
        for (auto ptr : *this) {
            delete ptr;
        }
    }
};

template<typename T>
class unref_ptr {
public:
    T * ptr;
    unref_ptr() {
        ptr = NULL;
    }
    unref_ptr( T * ptr ) {
        this->ptr = ptr;
    }
    void reset() {
        if ( ! ptr )
            return;
        unref( ptr );
        ptr = NULL;
    }
    ~unref_ptr() {
        reset();
    }
    unref_ptr<T> & operator=(T * ptr) {
        if ( ptr == this->ptr )
            return *this;
        reset();
        this->ptr = ptr;
        return *this;
    }
    T & operator*() {
        return *ptr;
    }
    T * operator->() {
        return ptr;
    }
    operator T *() {
        return ptr;
    }
    operator const T *() const {
        return ptr;
    }
    unref_ptr( const unref_ptr<T>& ) = delete; 
    unref_ptr<T> & operator=( unref_ptr<T>& ) = delete;
};
