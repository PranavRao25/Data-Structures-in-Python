from collections import deque
import math
import numpy as np
import random

class BinaryTree:
    class _Node:
        __slots__ = ['element', 'left', 'right', 'parent']

        def __init__(self, element: int, left = None, right=None, parent=None):
            self.element = element
            self.parent = parent
            self.left, self.right = left, right
    
    class Position:
        def __init__(self, container, node):
            self.container, self.node= container, node
        
        def element(self):
            return self.node.element
        
        def __eq__(self, other):
            return (self.container is other.container) and (self.node is other.node)

    def position_to_node(self, pos):
        if not isinstance(pos, self.Position): raise TypeError('Invalid type')
        elif pos is None:   raise ValueError('Null Position')
        # elif pos.container is not self: raise ValueError('Invalid position')
        elif pos.node.parent is pos.node:   raise ValueError('Redacted Node')
        else:   return pos.node
    
    def node_to_position(self, node):
        if not isinstance(node, self._Node): raise TypeError('Invalid type')
        elif node is None:  raise ValueError('Null Node')
        elif node.parent is node:   raise ValueError('Redacted Node')
        else:   return self.Position(container=self, node=node)

    def __init__(self):
        self._root = None
        self.__size = 0
    
    def add_root(self, val: int)->int:
        if self._root: raise Exception('Root already exists')
        self._root = self.node_to_position(self._Node(element=val))
        self.__size += 1
        return self._root
    
    def root(self):
        return self._root
    
    def __len__(self)->int:
        return self.__size
    
    def add_left(self, val, parent_pos):
        parent = self.position_to_node(parent_pos)
        if parent.left: raise Exception('Left child already exists')
        node = self._Node(element=val, parent=parent)
        parent.left = node
        self.__size += 1
        return self.node_to_position(node)
        
    def add_right(self, val, parent_pos):
        parent = self.position_to_node(parent_pos)
        if parent.right: raise Exception('Left child already exists')
        node = self._Node(element=val, parent=parent)
        parent.right = node
        self.__size += 1
        return self.node_to_position(node)

    def remove_node(self, pos):
        node = self.position_to_node(pos)
        
        if node.left and node.right:    raise Exception('Cannot Remove, both childern exists')
        
        if node.left:   node.left.parent = node.parent
        else:   node.right.parent = node.parent
        
        if node.parent.left is node:
            node.parent.left = node.left if node.left else node.right
        elif node.parent.right is node:
            node.parent.right = node.right if node.right else node.left
        else:   raise Exception('Invalid node')

        node.parent = node

    def replace(self, old_pos, new_pos):
        old, new = self.position_to_node(old_pos), self.position_to_node(new_pos)

        if old.parent.left is old:
            old.parent.left = new
        else:
            old.parent.right = new

        if new.parent.left is new:
            new.parent.left = old
        else:
            new.parent.right = old
        
        if old.left:
            old.left.parent = new
        if old.right:
            old.right.parent = new
        if new.left:
            new.left.parent = old
        if new.right:
            new.right.parent = old
        
        old.parent, new.parent = new.parent, old.parent
        old.left, new.left = new.left, old.left
        old.right, new.right = new.right, old.right

        return self.node_to_position(old), self.node_to_position(new)
    
    def left(self, pos):
        node = self.position_to_node(pos)
        return node.left
    
    def right(self, pos):
        node = self.position_to_node(pos)
        return node.right
    
    def parent(self, pos):
        node = self.position_to_node(pos)
        return node.parent
    
    def is_empty(self):
        return self.__size == 0

    def visit(self, node):
        print(node.element)
    
    def pre_order(self):
        def recurse_pre_order(node):
            if node:
                yield node
                for ch in recurse_pre_order(node.left):
                    yield ch
                for ch in recurse_pre_order(node.right):
                    yield ch
        
        if self._root:
            for node in recurse_pre_order(self.position_to_node(self._root)):
                yield node
    
    def post_order(self):
        def recurse_post_order(node):
            if node:
                for ch in recurse_post_order(node.left):
                    yield ch
                for ch in recurse_post_order(node.right):
                    yield ch
                yield node
        
        if self._root:
            for node in recurse_post_order(self.position_to_node(self._root)):
                yield node

    def in_order(self):
        def in_order_recur(node):
            if node.left:
                for child in in_order_recur(node.left):
                    yield child
            yield node
            if node.right:
                for child in in_order_recur(node.right):
                    yield child

        if self._root:
            for p in in_order_recur(self.position_to_node(self._root)):
                yield p
        
    def attach(self, pos, T1, T2):
        if not pos: raise ValueError('Null position')
        
        node = self.position_to_node(pos)
        if node.left and node.right:    raise ValueError('Node must not have childern')
        
        if T1:
            if not isinstance(T1, BinaryTree):  raise TypeError('Binary Trees or its derivatives')
            left_root = T1.position_to_node(T1.root())
            node.left = left_root
            left_root.parent = node
        if T2:
            if not isinstance(T2, BinaryTree):  raise TypeError('Binary Trees or its derivatives')
            right_root = T2.position_to_node(T2.root())
            node.right = right_root
            right_root.parent = node
        
        return self.node_to_position(node)

    def is_leaf(self, pos):
        node = self.position_to_node(pos)
        return not node.left and not node.right
    
    def num_childern(self, pos):
        node = self.position_to_node(pos)
        return (node.left is not None) + (node.right is not None)

    def is_root(self, p):
        return p is self._root
    
    def childern(self, p):
        if p:
            if p.left:
                yield p.left
            if p.right:
                yield p.right
    
    def height(self, p=None):
        def recur_height(p):
            if self.is_leaf(p): return 0
            else:   return 1 + max(recur_height(c) for c in self.childern(p))
    
        if not p:   p = self._root
        return recur_height(p)

    def depth(self):
        def recur_depth(arr, node):
            if node:
                if node is self._root:  arr[node] = 0
                if node.left:
                    arr[node.left] = arr[node] + 1
                    recur_depth(arr,node.left)
                if node.right:
                    arr[node.right] = arr[node] + 1
                    recur_depth(arr, node.right)
        
        arr = dict()
        recur_depth(arr, self.position_to_node(self._root))
        return arr
    
    def level_order(self):
        temp, queue = self.position_to_node(self._root), deque()
        visited = {temp : True}

        queue.append(temp)
        while len(queue):
            node = queue.popleft()
            print(node.element)
            visited[node] = True

            for c in self.childern(node):
                if c not in visited:
                    queue.append(c)

    def path_length(self):
        """The path length of a tree T is the sum of the depths of all positions in T ."""
        temp, queue = self.position_to_node(self._root), deque()
        length, depth = 0, 0
        visited = {temp : True}

        queue.append((depth, temp))
        while len(queue):
            depth, node = queue.popleft()
            length += depth
            visited[node] = True

            for c in self.childern(node):
                if c not in visited:
                    queue.append((depth + 1, c))
        return length

def preorder_next(T, p):
    if T.left(p):    return T.node_to_position(T.left(p))
    elif T.right(p): return T.node_to_position(T.right(p))
    else:   return None

def postorder_next(T, p):
    if T.is_root(p):    return None
    else:   return T.node_to_position(T.parent(p))

def in_order(T, p):
    if T.right(p):  return T.node_to_position(T.right(p))
    else:   return None

def max_depth_tree(T, pos):
    """Find the max_depth of the tree"""

    def recur(node, depth, max_depth):
        if node.left:
            if max_depth[0] < (depth + 1):   max_depth[0] = (depth + 1)
            recur(node.left, depth + 1, max_depth)
        if node.right:
            if max_depth[0] < (depth + 1):   max_depth[0] = (depth + 1)
            recur(node.right, depth + 1, max_depth)

    temp, depth = T.position_to_node(pos), 0
    max_depth = [0]
    recur(temp, depth, max_depth)
    return max_depth[0] + 1

def diameter(T):
    """
    The diameter of T is the maximum distance between two positions in T .
    The two positions have to be on opposite ends of the tree i.e LCA(p, q) = root
    They also need to have the maximum depth in their respective subtrees.
    """

    root = T.root()
    left, right = T.left(root), T.right(root)

    return max_depth_tree(T, left) + max_depth_tree(T, right) + 1

def last_common_ancestor(T, p, q):
    """the lowest common ancestor (LCA) between two positions p and q as the lowest position in T that has
       both p and q as descendants (where we allow a position to be a descendant of itself )."""
    
    temp1, temp2 = T.position_to_node(p), T.position_to_node(q)

    while temp1 and temp2:
        if temp1 == temp2:
            return temp1
        temp1, temp2 = temp1.parent, temp2.parent
    return None

def reflection(T):
    """the reﬂection of T to be the binary tree T' such that each node v in T is also in T',
       but the left child of v in T is v’s right child in T' and the right child of v in T
       is v’s left child in T'."""
    
    def recur(node1, node2):
        if node1.left:
            T_r.add_right(node1.left.element, T_r.node_to_position(node2))
            recur(node1.left, node2.right)
        if node1.right:
            T_r.add_left(node1.right.element, T_r.node_to_position(node2))
            recur(node1.right, node2.left)

    T_r = BinaryTree()
    T_r.add_root(T.root().element())
    recur(T.position_to_node(T.root()), T_r.position_to_node(T_r.root()))
    return T_r

def two_pointer_tree_traversal(T):
    now, old = T.root(), None
    
    if now is None: raise Exception("Tree Empty")
    while True:
        print(now.val)
        if T.leaf(now): now, old = now.parent, now
        elif now.parent == old:   now, old = now.left, now
        elif now.left == old:   now, old = now.right, now
        elif now.right == old:  now, old = now.parent, now
        elif now is None:
            if old == T.root(): break
            now, old = old, now

def bfs(tree, pos, cond):
    temp, queue = tree.position_to_node(pos), deque()
    visited = {temp:True}

    queue.append(temp)
    while len(queue):
        node = queue.popleft()
        if cond(node):
            return node
        visited[node] = True

        if node.left and not visited[node]:
            queue.append(node)
        if node.right and not visited[node]:
            queue.append(node)
    else:
        return None

def count_leaves(tree):
    # method 1
    return len(tree) + 1 - 2**(math.floor(math.log2(len(tree))))
    
    # method 2
    # temp, queue, count = tree.root(), deque(), 0
    # visited = {tree.position_to_node(temp):True}

    # queue.append(temp)
    # while len(queue):
    #     pos = queue.popleft()
    #     if tree.is_leaf(pos):
    #         count += 1
        
    #     node = tree.position_to_node(pos)
    #     visited[node] = True
    #     if node.left and not visited[node]:
    #         queue.append(pos)
    #     if node.right and not visited[node]:
    #         queue.append(pos)
    # else:
    #     return count    

class BinarySearchTree(BinaryTree):
    def add(self, val, parent_pos=None):
        if parent_pos:
            parent_element = parent_pos.element()
            return self.add_right(val, parent_pos=parent_pos
                ) if parent_element < val else self.add_left(val, parent_pos=parent_pos)
        else:  # manual
            temp, node = self.position_to_node(self._root), self._Node(element=val)
            while temp:
                curr_val = temp.element
                if val < curr_val:  # go to left
                    if temp.left is None:
                        temp.left, node.parent = node, temp
                        break
                    else:
                        temp = temp.left
                else:
                    if temp.right is None:
                        temp.right, node.parent = node, temp
                        break
                    else:
                        temp = temp.right
            else:
                return self.node_to_position(node)
    
    def balance(self):
        def build_balanced_tree(nodes, begin, end)->BinaryTree:
            if begin > end: return None
            
            mid = (begin + end) // 2
            
            sub_tree = BinaryTree()
            root_pos = sub_tree.add_root(nodes[mid].element)
            sub_tree.attach(
                root_pos,
                build_balanced_tree(nodes, begin, mid - 1),
                build_balanced_tree(nodes, mid + 1, end)
                )
            return sub_tree
        
        l = list(self.in_order())
        new_tree = build_balanced_tree(l, 0, len(l) - 1)
        return new_tree

    def remove(self, val):
        temp = self.position_to_node(self._root)
        while temp:
            curr = temp.element

            if curr == val:
                if temp.left:
                    temp.left.parent, temp.parent.right = temp.parent, temp.left
                    temp.left.right, temp.right.parent = temp.right, temp.left
                    temp.parent, temp.left, temp.right = temp, temp, temp
                elif temp.right:
                    temp.right.parent, temp.parent.right = temp.parent, temp.right
                    temp.parent, temp.right = temp, temp
                else:
                    temp.parent.right = None
                    temp.parent = None
                break
            elif curr < val:
                temp = temp.right
            else:
                temp = temp.left
        else:
            raise Exception('Didn\'t find node')
        return self.balance()

def build_expression_tree(expression: str):
    if not isinstance(expression, str): raise TypeError('expression has to be a string')
    stack, tokens = deque(), []
    
    i = 0
    while i < len(expression):
        char = expression[i]
        if char.isdecimal():
            s = str(char)
            while i < len(expression) - 1:
                char = expression[i+1]

                if not char.isdecimal():
                    break
                else:
                    s += char
                i += 1
            tokens.append(s)
        elif char in ['(', ')', '/', '+', '-', '*']:
            tokens.append(char)
        elif char.isspace():    pass
        else:
            raise Exception('Illegal character')
        i += 1

    for token in tokens:
        if token == '(': stack.append(token)
        elif token.isdecimal():
            node = BinaryTree()
            _ = node.add_root(int(token))
            stack.append(node)
        elif token in ['/', '+', '-', '*']:
            stack.append(token)
        elif token == ')':
            if len(stack) < 3:  raise Exception('not enough tokens')
            right_tree = stack.pop()
            operator = stack.pop()
            left_tree = stack.pop()

            operator_node = BinaryTree()
            op_pos = operator_node.add_root(operator)
            _ = operator_node.attach(op_pos, left_tree, right_tree)
            paren = stack.pop()

            if paren != '(':   raise ValueError('Missing bracket')

            stack.append(operator_node)
        else:   raise ValueError('Illegal Token')
    else:
        if len(stack) != 1: raise Exception('Incorrect Operation')
        exp_tree = stack.pop()
        return exp_tree

def evaluate_expression_tree(tree : BinaryTree):
    def recurse_eval(node):
        if not node.left and not node.right:
            if not isinstance(node.element, int):   raise TypeError('Operands has to be int')
            return int(node.element)
        operator = node.element
        left_operand, right_operand = recurse_eval(node.left), recurse_eval(node.right)

        match(operator):
            case '*':   return left_operand * right_operand
            case '/':   return left_operand / right_operand
            case '+':   return left_operand + right_operand
            case '-':   return left_operand - right_operand
            case _: raise Exception('Invalid operator')

    return recurse_eval(tree.position_to_node(tree.root()))

def tree_to_expression(tree):
    def in_order(node, result):
        if node:
            result.append('(')
            in_order(node.left, result)
            result.append(str(node.element))
            in_order(node.right, result)
            result.append(')')
    
    result = []
    in_order(tree.position_to_node(tree.root()), result)
    return ''.join(result)

class ArrayBinaryTree:
    def __init__(self, val):
        self._array = [val, None, None]
        self._size = 1
        self._height = 0
    
    def add_left(self, val, pos):
        if pos >= self._size:   raise ValueError('Invalid Position '+ pos)
        new_position = 2 * pos + 1
        
        if new_position < len(self._array) and self._array[new_position]:
            raise ValueError('Left child already exists')
        
        self._size += 1
        if new_position >= len(self._array):
            self._array.extend(
                [None] * int(2 ** (self._height + 1))
                )
        self._height = int(math.log2(len(self._array) + 1) - 1)
        self._array[new_position] = val
        return new_position

    def add_right(self, val, pos):
        if pos >= self._size:   raise ValueError('Invalid Position')
        new_position = 2 * pos + 2
        
        if new_position < len(self._array) and self._array[new_position]:
            raise ValueError('Right child already exists')

        self._size += 1
        if new_position >= len(self._array):
            self._array.extend(
                [None] * int(2 ** (self._height + 1))
                )
        self._height = int(math.log2(len(self._array) + 1) - 1)
        self._array[new_position] = val
        return new_position
    
    def remove(self, pos):
        if pos >= self._size:   raise ValueError('Invalid Position')
        val = self._array[pos]
        self._array[pos] = None
        self._size -= 1
        
        k = 0
        for i in range(len(self._array) - 1, -1, -1):
            if self._array[i] is None:  k += 1
        if k >= 2**self._height:
            for i in range(2**self._height):
                self._array.pop()
            self._height -= 1
        return val

    def is_leaf(self, pos):
        if (2 * pos + 1) >= len(self._array):
            return True
        else:
            if (2 * pos + 2) >= len(self._array):   return self._array[2 * pos + 1] is None
            return self._array[2 * pos + 1] is None and self._array[2 * pos + 2] is None

    def root(self):
        return self._array[0]
    
    def num_childern(self, pos):
        if (2 * pos + 1) >= len(self._array):   return 0
        elif (2 * pos + 2) >= len(self._array): return 1
        else:   return 2

    def attach(self, pos, T1, T2):
        if pos >= self._size:   raise ValueError('Invalid Position')
        
        self._height += max(T1.height(), T2.height())
        self._array.extend([None] * (2 ** (max(T1.height(), T2.height()) + 1)))

        left_offset = 2 * pos + 1
        if left_offset < len(self._array) and not self._array[left_offset]:
            for p,v in T1.in_order():
                self._array[left_offset + p] = v
        else:   raise ValueError('Left child already exists')
        right_offset = 2 * pos + 2
        if right_offset < len(self._array) and not self._array[right_offset]:
            for p,v in T1.in_order():
                self._array[right_offset + p] = v
        else:   raise ValueError('Right child already exists')                

    def pre_order(self):
        def recurse_pre_order(pos, result):
            if pos < self._size:
                result.append((pos, self._array[pos]))
                recurse_pre_order(2 * pos + 1, result)
                recurse_pre_order(2 * pos + 2, result)

        result = []
        recurse_pre_order(0, result)
        return result

    def post_order(self):
        def recurse_post_order(pos, result):
            if pos < self._size:
                recurse_post_order(2 * pos + 1, result)
                recurse_post_order(2 * pos + 2, result)
                result.append((pos, self._array[pos]))

        result = []
        recurse_post_order(0, result)
        return result

    def in_order(self):
        def recurse_in_order(pos, result):
            if pos < self._size:
                recurse_in_order(2 * pos + 1, result)
                result.append((pos, self._array[pos]))
                recurse_in_order(2 * pos + 2, result)

        result = []
        recurse_in_order(0, result)
        return result

    def __len__(self):
        return len(self._array)
    
    def is_empty(self):
        return self._size == 0

    def height(self):
        return self._height

class AVLTree(BinarySearchTree):
    pass

def minimax(T, node, depth, maximising_player):
    if depth == 0:  return T.eval(node)
    
    k = max(minimax(T, child, depth - 1, not maximising_player) 
            for child in T.childern(node)) if maximising_player else min(
                minimax(T, child, depth - 1, not maximising_player) 
            for child in T.childern(node))
    return k

def alpha_beta(T, node, depth, alpha, beta, maximising_player):
    if depth == 0:  return T.eval(node)

    if maximising_player:
        max_eval = np.inf()
        for child in T.childern(node):
            eval = minimax(T, child, depth - 1, alpha, beta, not maximising_player)
            max_eval = max(eval, max_eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = - np.inf()
        for child in T.childern(node):
            eval = minimax(T, child, depth - 1, alpha, beta, not maximising_player)
            min_eval = max(eval, min_eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

class Heap(BinaryTree):
    def add_node(self, val):
        temp, node = self.position_to_node(self._root), self._Node(element=val)
        
        if temp.element <= val:
            node.left, temp.parent = temp, node
            self._root = self.node_to_position(node)
        else:
            while temp:
                curr_val = temp.element

                if curr_val <= val:
                    node.left, node.parent = temp, temp.parent
                    
                    if temp == temp.parent.left:    temp.parent.left = node
                    if temp == temp.parent.right:   temp.parent.right = node
                    temp.parent = node
                else:
                    if temp.left is None:
                        self.add_left(val, self.node_to_position(temp))
                        break
                    if temp.right is None:
                        self.add_right(val, self.node_to_position(temp))
                        break
                    temp = temp.left if np.random.choice(1) == 0 else temp.right

    def remove_node(self, val):
        temp, queue = self.position_to_node(self._root), deque()
        visited = {temp:True}

        queue.append(temp)
        while len(queue):
            node = queue.popleft()
            if node.element == val:
                break
            visited[node] = True

            if node.left and not visited[node]:
                queue.append(node)
            if node.right and not visited[node]:
                queue.append(node)
        else:   raise ValueError("Node not found")

        if node.left > node.right:
            node.left.parent = node.parent
            if node == node.parent.left: node.parent.left = node.left
            if node == node.parent.right: node.parent.right = node.left
            node.right.parent = node.left
        else:
            node.right.parent = node.parent
            if node == node.parent.left: node.parent.left = node.right
            if node == node.parent.right: node.parent.right = node.right
            node.left.parent = node.right
        
        node.parent = node.left = node.right = None
        