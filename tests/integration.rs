use kruma::tensor::{Tensor, ops::TensorOps};

#[test]
fn test_add() {
    let mut a = Tensor::<f32, 2>::new([2, 2]);
    a[[0, 0]] = 1.0;
    a[[0, 1]] = 2.0;

    let mut b = Tensor::<f32, 2>::new([2, 2]);
    b[[0, 0]] = 3.0;
    b[[0, 1]] = 4.0;

    let c = a.add(&b);
    assert_eq!(c[[0, 0]], 4.0);
    assert_eq!(c[[0, 1]], 6.0);
}

#[test]
fn test_mul() {
    let mut a = Tensor::<f32, 2>::new([2, 2]);
    a[[0, 0]] = 1.0;
    a[[0, 1]] = 2.0;
    a[[1, 0]] = 3.0;
    a[[1, 1]] = 4.0;

    let mut b = Tensor::<f32, 2>::new([2, 2]);
    b[[0, 0]] = 5.0;
    b[[0, 1]] = 6.0;
    b[[1, 0]] = 5.0;
    b[[1, 1]] = 6.0;


    let c = a.mul(&b);
    assert_eq!(c[[0, 0]],5.0);
    assert_eq!(c[[0,1]],12.0);
    assert_eq!(c[[1,0]],15.0);
    assert_eq!(c[[1,1]],24.0)
}

#[test]
fn test_matmul() {
    let mut a = Tensor::<f32, 2>::new([2, 2]);
    a[[0, 0]] = 1.0;
    a[[0, 1]] = 2.0;
    a[[1, 0]] = 3.0;
    a[[1, 1]] = 4.0;    

    let mut b = Tensor::<f32, 2>::new([2, 2]);
    b[[0, 0]] = 5.0;
    b[[0, 1]] = 6.0;
    b[[1, 0]] = 7.0;
    b[[1, 1]] = 8.0;

    let c = a.matmul(&b);
    assert_eq!(c[[0, 0]], 1.0 * 5.0 + 2.0 * 7.0); // 19
    assert_eq!(c[[0, 1]], 1.0 * 6.0 + 2.0 * 8.0); // 22
    assert_eq!(c[[1, 0]], 3.0 * 5.0 + 4.0 * 7.0); // 43
    assert_eq!(c[[1, 1]], 3.0 * 6.0 + 4.0 * 8.0); // 50
}

#[test]
fn test_relu() {
    let mut a = Tensor::<f32, 2>::new([2, 2]);
    a[[0, 0]] = -1.0;
    a[[0, 1]] = 2.0;
    a[[1, 0]] = -3.0;
    a[[1, 1]] = 4.0;  

    let c =a.relu();
    assert_eq!(c[[0, 0]], 0.0);
    assert_eq!(c[[0, 1]], 2.0);
    assert_eq!(c[[1, 0]], 0.0);
    assert_eq!(c[[1, 1]], 4.0);
}
