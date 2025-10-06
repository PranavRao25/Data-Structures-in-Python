from collections import deque
import math
import random

class StackinArray:
    class SubStack:
        def __init__(self, head, direction):
            self.head = head
            self.size = 0
            self.direction = direction
        
        def push(self, val)->int:
            if self.direction:  self.head -= 1
            else:   self.head += 1
            self.size += 1
            return self.head
        
        def pop(self):
            if self.size <= 0:  raise Exception(f'Stack {int(self.direction)} empty')
            t = self.head
            if self.direction:  self.head -= 1
            else:   self.head += 1
            self.size -= 1
            return t

        def top(self):
            return self.head
    
    def __init__(self, max_len):
        self.array = [None] * max_len
        self.max_len = max_len
        self.stacks = [StackinArray.SubStack(-1, False), StackinArray.SubStack(max_len, True)]
                    
    def is_full(self):
        return self.stacks[0].size + self.stacks[1].size >= self.max_len
    
    def push(self, stack_index, val):
        if self.is_full():
            raise Exception('Array full')
        new_index = self.stacks[stack_index].push(val)
        self.array[new_index] = val

    def pop(self, stack_index):
        new_index = self.stacks[stack_index].pop()
        t = self.array[new_index]
        self.array[new_index] = None
        return t
    
    def top(self, stack_index):
        return self.array[self.stacks[stack_index].top()]

class DoubleLinkedList:
    class _Node:
        __slots__ = ['element', 'index', 'next', 'previous']

        def __init__(self, element, index, next=None, previous=None):
            self.element = element
            self.index = index
            self.next = next
            self.previous = previous
    
    def __init__(self, val, index=0):
        self.__head = self._Node(element=val, index=index)
        self.__head.previous = self.__head
        self.__head.next = self.__head
        self.__tail = self.__head

        self.length = 1
    
    def insert_node(self, val, parent_index):
        if parent_index > self.length // 2:
            temp = self.__tail
            
            while temp.previous != self.__tail:
                if temp.index == parent_index:
                    break
                temp = temp.previous
            node = self._Node(
                element=val, index=parent_index+1,
                next=temp.next, previous=temp
                )
            temp.next.previous = node
            temp.next = node
            if temp == self.__tail:
                self.__tail, self.__head = node, node.next
        else:
            temp = self.__head
            while temp.next != self.__head:
                if temp.index == parent_index:
                    break
                temp = temp.next
            node = self._Node(
                element=val, index=parent_index+1,
                next=temp.next, previous=temp
                )
            temp.next.previous = node
            temp.next = node
            if temp == self.__head:
                self.__head, self.__tail = node, node.previous

        self.length += 1
        return node.index
    
    def head(self):
        return self.__head
    
    def tail(self):
        return self.__tail

    def print(self):
        temp = self.__head

        print(f"head, {self.head().element}")
        while True:
            print(temp.index, temp.element)
            temp = temp.next

            if temp == self.__head:
                break
        print(f"tail, {self.tail().element}")

    def remove_node(self, index):
        if index > self.length // 2:
            temp = self.__tail

            while True:
                if temp.index == index or temp == self.__head:
                    break
                temp = temp.previous
            (temp.parent).next, (temp.next).parent = temp.next, temp.parent
            temp.next, temp.parent = temp, temp
        else:
            temp = self.__head

            while True:
                if temp.index == index or temp == self.__tail:
                    break
                temp = temp.next
            (temp.parent).next, (temp.next).parent = temp.next, temp.parent
            temp.next, temp.parent = temp, temp

class Stack:
    min_size = -1
    def __init__(self, max_size = 100):
        self.stack = []
        self.curr_size = self.min_size
        self.max_size = max_size
    
    def top(self):
        return self.stack[self.curr_size]

    def push(self, val):
        if self.curr_size >= self.max_size:
            raise Exception('Stack Overflow')
        else:
            self.stack.append(val)
            self.curr_size += 1

    def pop(self):
        if self.is_empty():
            raise Exception('Stack Underflow')
        else:
            self.curr_size -= 1
            return self.stack.pop()

    def sequential_empty(self):
        if self.curr_size == -1:
            return
        self.stack.pop()
        self.curr_size -= 1
        self.sequential_empty()

    def is_empty(self):
        return self.curr_size == self.min_size

def signature_transfer(S: Stack, T: Stack):
    while True:
        try:
            T.push(S.pop())
        except Exception as e:
            break

def reverse_list(L):
    S = Stack()

    for i in L:
        S.push(i)
    rev = []
    while True:
        try:
            rev.append(S.pop())
        except Exception as e:
            break
    return rev

# S : T->
# t1: T->
# t2: T-> 0 1 2 3 4 5 6 7 8 9

def reverse_stack(S):
    # reverse the stack in place
    t1, t2 = Stack(), Stack()

    while True:
        try:
            t1.push(S.pop())
        except Exception as e:
            break
    
    while True:
        try:
            t2.push(t1.pop())
        except Exception as e:
            break
    
    while True:
        try:
            S.push(t2.pop())
        except Exception as e:
            break            

def enumerate_permutations_stack(l):
    permutations, stack = [], Stack()
    
    """
    We are essentially going to parse the permutation tree
    Each node in the tree is a subset of the complete set of n numbers with the following property:
    1. len(child) = len(node) + 1 for all child in node.childern
    2. # childern in level i = # childern in level (i-1) - 1
    
    The final leaves of the tree contains all of the permutations.
    As we are essentially traversing this hypothetical tree, we can use either stack (DFS) or queue (BFS)
    Benefit of stack is we can effectively maintain the stack size without overflow.
    Space Complexity using Stack is O(n2)
    """

    n = len(l)
    stack.push((
        [],  # current_partial_perms
        [False] * n,    # which elements are present
        0   # pos of perm
    ))
    while not stack.is_empty():
        curr_perm, used, pos = stack.pop()

        if pos == n:    # level n means leaf node
            permutations.append(curr_perm)
            continue

        for i in range(n-1, -1, -1):  # childern of the current node of partial perms / subset
            if not used[i]:
                new_perm = curr_perm + [l[i]]
                new_used = used[:]
                new_used[i] = True
                new_pos = pos + 1   # moving to one level down
                stack.push((new_perm, new_used, new_pos))
    return permutations

class Queue:
    def __init__(self, max_size = 100):
        self.queue = []
        self.min_size = -1
        self.curr_size = self.min_size
        self.max_size = max_size
    
    def enqueue(self, val):
        # if self.curr_size >= self.max_size:
        #     raise Exception('Queue Overflow')
        
        self.queue.append(val)
        self.curr_size += 1

    def dequeue(self):
        if self.is_empty():
            raise Exception('Queue Underflow')
        
        self.curr_size -= 1
        return self.queue.pop(0)

    def front(self):
        return self.queue[0]

    def back(self):
        return self.queue[-1]

    def is_empty(self):
        return self.curr_size == self.min_size

def enumerate_permutations_queue(l):
    permutations, queue = [], Queue()
    
    """
    We are essentially going to parse the permutation tree
    Each node in the tree is a subset of the complete set of n numbers with the following property:
    1. len(child) = len(node) + 1 for all child in node.childern
    2. # childern in level i = # childern in level (i-1) - 1
    
    The final leaves of the tree contains all of the permutations.
    As we are essentially traversing this hypothetical tree, we can use either stack (DFS) or queue (BFS)
    Using Queue will have space overflow issues.
    Space Complexity O(n!)
    """

    n = len(l)
    queue.enqueue((
        [],  # current_partial_perms
        [False] * n,    # which elements are present
        0   # pos of perm
    ))
    while not queue.is_empty():
        curr_perm, used, pos = queue.dequeue()

        if pos == n:    # level n means leaf node
            permutations.append(curr_perm)
            continue

        for i in range(n):  # childern of the current node of partial perms / subset
            if not used[i]:
                new_perm = curr_perm + [l[i]]
                new_used = used[:]
                new_used[i] = True
                new_pos = pos + 1   # moving to one level down
                queue.enqueue((new_perm, new_used, new_pos))
    return permutations

class StackADT:
    min_size = -1

    def __init__(self, max_size = 100):
        self.stack = Queue()    
        self.curr_size = self.min_size
        self.max_size = max_size
    
    def push(self, val):  # O(n) top of the stack is at the front of the queue
        if self.curr_size >= self.max_size:
            raise Exception('Stack Overflow')
        else:
            self.stack.enqueue(val)
            while self.stack.front() != val:
                self.stack.enqueue(self.stack.dequeue())
            else:
                self.curr_size += 1

    def pop(self):  # O(1)
        if self.is_empty():
            raise Exception('Stack Underflow')
        
        self.curr_size -= 1
        return self.stack.dequeue()

    def top(self):
        return self.stack.front()

    def is_empty(self):
        return self.curr_size == self.min_size

class QueueADT:
    min_size = -1

    def __init__(self, max_size = 100):
        # one stack always remains empty
        self.max_size = max_size
        self.curr_size = self.min_size
        self.tail = None
        self.head = None
        self.stacks = (Stack(max_size=self.max_size), 
                       Stack(max_size=self.max_size))
        self.front_stack = 0  # which stack to pop from
        self.back_stack = 1  # which stack to push to
    
    def enqueue(self, val):
        # push element to back stack
        if self.curr_size >= self.max_size:
            raise Exception("Queue Overflow")
        
        self.stacks[self.back_stack].push(val)
        self.curr_size += 1

    def dequeue(self):
        if self.is_empty():
            raise Exception('Queue Underflow')
        
        val = self.stacks[self.front_stack].pop()
        

    def front(self):
        pass

    def back(self):
        pass

    def is_empty(self):
        return self.curr_size == self.min_size

def search_stack(S, x):
    Q = Queue(max_size=S.max_size)

    while not S.is_empty():
        t = S.pop()

        if t == x:
            while not Q.is_empty():
                S.push(Q.dequeue())
            return True
        else:
            Q.enqueue(t)
            while Q.front() != t:
                Q.enqueue(Q.dequeue())
    return False

class Node:
    __slots__ = ["val", "next"]

    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self, val):
        self.head = Node(val)

    def append(self, val):
        node = Node(val)
        t = self.head
        
        while t.next:
            t = t.next
        t.next = node

    def insert(self, index, val):
        t, node, i = self.head, Node(val), 0
        
        while t.next:
            if i == index:
                # insertion
                node.next = t.next
                t.next = node
                break
            t = t.next
            i += 1
        else:
            if i <= index:
                print("incorrect index")
            else:
                print("Index Not found")
    
    def push(self, val):
        node = Node(val)
        node.next = self.head
        self.head = node

    def search(self, val):
        t = self.head

        while t:
            if t.val == val:
                return t
            t = t.next
        return

class ListNode:
    __slots__ = ("val", "_next")

    def __init__(self, x):
        self.val = x
        self._next = None
    
    def __eq__(self, node):
        return self.val == node.val


def type_check(value):
    if not isinstance(value, int):
            raise TypeError("Only Integer Linked Lists allowed")


class LinkedList(ListNode):
    def __init__(self, head_value):
        try:
            type_check(head_value)
            self.head = ListNode(head_value)
            self.__last_node = self.head
        except:
            print("Linked list cannot be created")
    
    def find(self, node, op):
        # op = True if insert else sub
        try:
            temp = self.head

            if temp.val == node.val:
                print("Value is the head")
                return temp
            while temp._next:
                k = temp._next
                if k.val == node.val:
                    return k if op else temp
                temp = temp._next
            print("Value couldn't be found")
        except:
            print("Value couldn't be found")
    
    def append(self, value):
        try:
            type_check(value)
            node = ListNode(value)
            self.__last_node._next = node
            self.__last_node = node
        except:
            print("Could not be added")
    
    def push(self, value):
        try:
            type_check(value)
            node = ListNode(value)
            node._next = self.head
            self.head = node
        except:
            print("Could not be added")

    def insert(self, value, head):
        try:
            node = ListNode(value)
            parent = self.find(ListNode(head), op=True)
            node._next = parent._next
            parent._next = node
        except:
            print("Value couldn't be inserted")

    def remove(self, value):
        try:
            type_check(value)
            node = ListNode(value)

            if self.head == node:
                print("You have deleted the List")
                self.head = None
            else:
                parent = self.find(node, op=False)
                parent._next = (parent._next)._next
        except:
            print("Value couldn\'t be subtracted")
    
    def __str__(self):
        temp = self.head
        list_rep = str(temp.val)
        
        temp = temp._next
        while temp:
            list_rep += "->" + str(temp.val)
            temp = temp._next
        else:
            return list_rep

def second_last_node(ll):
    t = ll.head

    if t._next is None:
        return t.val
    else:
        while ((t._next)._next):
            t = t._next
        else:
            return t.val

def count_linked_list(ll):
    if ll is None:
        return 0
    return 1 + count_linked_list(ll._next)

def swap_nodes(ll, a, b):
    p_a, p_b = ll.find(ListNode(a), op=False), ll.find(ListNode(b), op=False)
    m, n = p_a._next, p_b._next

    p_a._next, p_b._next = n, m
    m._next, n._next = n._next, m._next

