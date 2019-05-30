A = rand(6, 10);
display( A);

[U, S, V]  = svd(A);

display(U);
display(S);
display(V);

Apseudo = pinv(A);
display(Apseudo);
