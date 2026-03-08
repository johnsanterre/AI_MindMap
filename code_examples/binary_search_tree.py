"""Binary Search Tree — insert, search, in-order traversal."""

class Node:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None

def insert(root, val):
    if root is None:
        return Node(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    return root

def search(root, val):
    if root is None:
        return False
    if val == root.val:
        return True
    return search(root.left if val < root.val else root.right, val)

def inorder(root):
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def height(root):
    if root is None:
        return 0
    return 1 + max(height(root.left), height(root.right))

# --- demo ---
root = None
for v in [5, 3, 7, 1, 4, 6, 8, 2]:
    root = insert(root, v)

print("In-order:", inorder(root))
print("Height:", height(root))
print("Search 4:", search(root, 4))
print("Search 9:", search(root, 9))
