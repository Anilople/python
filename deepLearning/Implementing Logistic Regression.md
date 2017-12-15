$$
\frac {dz}{dw_i}= x^{(i)}
$$

$$
\frac {dJ}{da}=-\frac {y^{i}}a + \frac {1-{y^{i}}}{1-a}
$$

$$
\frac{da}{dz}=a(1-a)
$$

$$
\frac {dz}{dw_i}= x^{(i)}
$$

$$
\frac{dz}{db}=1
$$



In code, we use notation like those.
$$
dz_i \longrightarrow \frac{dL}{dz_i} = \frac{dL}{da} \frac{da}{dz_i} = (-\frac {y^{i}}a + \frac {1-{y^{i}}}{1-a})*a(1-a) = a-y
$$

$$
dw_i \longrightarrow \frac{dL}{dw_i} =  \frac{dL}{dz_i}  \frac{dz_i}{dw_i} = (a-y) x^{(i)}
$$

$$
db \longrightarrow \frac{dL}{db} =  \frac{dL}{dz}  \frac{dz}{db} = a-y
$$

$J =0 , \ dw_i = 0, \ dw_2 = 0, \ db=0 \qquad (J=L)$

$for \quad i = 1 \quad to \quad m:$

$\qquad z^{(i)} = w^T x^{(i)}+b$

$\qquad a^{(i)} = \sigma(z^{(i)})$

$\qquad J+= - [ \ y^{(i)}\log a^{(i)} + (1-y^{(i)})\log(1-a^{(i)})]$

$\qquad dw_1 +=x_1^{(i)}dz^{(i)}$

$\qquad dw_2 +=x_2^{(i)}dz^{(i)}$

$\qquad db +=dz^{(i)}$

$J=J/m, \ dw_1=dw_1/m, \ dw_2=dw_2/m, \ db=db/m$

Like picture:

