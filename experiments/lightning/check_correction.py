"""Hand-calculated regression checks for SelfLift paper Eq. 8."""
import ast
from pathlib import Path


def check():
    import torch
    # Isolate the tensor math without importing the full renderer.
    tree=ast.parse((Path(__file__).parent/"selflift.py").read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="artifact_correct")
    namespace={"torch":torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"selflift.py","exec"),namespace)
    correct=namespace["artifact_correct"]
    direct=torch.zeros(1,1,1,1,4)
    anchor=torch.tensor([1.,2.,3.,4.]).reshape_as(direct)
    torch.testing.assert_close(correct(direct,anchor,rho=.5),torch.tensor([0.,0.,1.5,4.]).reshape_as(direct))
    torch.testing.assert_close(correct(direct,anchor,rho=0),direct)
    torch.testing.assert_close(correct(direct,torch.ones_like(direct),rho=1),torch.full_like(direct,.5))
    torch.testing.assert_close(correct(direct,direct),direct)
    print("ADAPTIVE_CORRECTION_CHECKS_PASSED",flush=True)


if __name__=="__main__":
    check()
