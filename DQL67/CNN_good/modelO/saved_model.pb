úõ
ý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02unknown8²¼

conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_21/kernel
~
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*'
_output_shapes
:*
dtype0
u
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_21/bias
n
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes	
:*
dtype0
~
dense_105/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
<*!
shared_namedense_105/kernel
w
$dense_105/kernel/Read/ReadVariableOpReadVariableOpdense_105/kernel* 
_output_shapes
:
<*
dtype0
u
dense_105/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_105/bias
n
"dense_105/bias/Read/ReadVariableOpReadVariableOpdense_105/bias*
_output_shapes	
:*
dtype0
~
dense_106/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_106/kernel
w
$dense_106/kernel/Read/ReadVariableOpReadVariableOpdense_106/kernel* 
_output_shapes
:
*
dtype0
u
dense_106/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_106/bias
n
"dense_106/bias/Read/ReadVariableOpReadVariableOpdense_106/bias*
_output_shapes	
:*
dtype0
}
dense_107/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namedense_107/kernel
v
$dense_107/kernel/Read/ReadVariableOpReadVariableOpdense_107/kernel*
_output_shapes
:	*
dtype0
t
dense_107/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_107/bias
m
"dense_107/bias/Read/ReadVariableOpReadVariableOpdense_107/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
Ì
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueýBú Bó
§
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
6
)iter
	*decay
+learning_rate
,momentum
 
8
0
1
2
3
4
5
#6
$7
8
0
1
2
3
4
5
#6
$7

-metrics
.non_trainable_variables
regularization_losses
		variables

trainable_variables
/layer_regularization_losses

0layers
 
\Z
VARIABLE_VALUEconv2d_21/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_21/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

1metrics
2non_trainable_variables
regularization_losses
	variables
trainable_variables
3layer_regularization_losses

4layers
 
 
 

5metrics
6non_trainable_variables
regularization_losses
	variables
trainable_variables
7layer_regularization_losses

8layers
\Z
VARIABLE_VALUEdense_105/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_105/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

9metrics
:non_trainable_variables
regularization_losses
	variables
trainable_variables
;layer_regularization_losses

<layers
\Z
VARIABLE_VALUEdense_106/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_106/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

=metrics
>non_trainable_variables
regularization_losses
 	variables
!trainable_variables
?layer_regularization_losses

@layers
\Z
VARIABLE_VALUEdense_107/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_107/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1

Ametrics
Bnon_trainable_variables
%regularization_losses
&	variables
'trainable_variables
Clayer_regularization_losses

Dlayers
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

E0
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	Ftotal
	Gcount
H
_fn_kwargs
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

F0
G1
 

Mmetrics
Nnon_trainable_variables
Iregularization_losses
J	variables
Ktrainable_variables
Olayer_regularization_losses

Players
 

F0
G1
 
 

serving_default_conv2d_21_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
µ
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_21_inputconv2d_21/kernelconv2d_21/biasdense_105/kerneldense_105/biasdense_106/kerneldense_106/biasdense_107/kerneldense_107/bias*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*0
f+R)
'__inference_signature_wrapper_180407278
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ú
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$dense_105/kernel/Read/ReadVariableOp"dense_105/bias/Read/ReadVariableOp$dense_106/kernel/Read/ReadVariableOp"dense_106/bias/Read/ReadVariableOp$dense_107/kernel/Read/ReadVariableOp"dense_107/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__traced_save_180407495
Ý
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_21/kernelconv2d_21/biasdense_105/kerneldense_105/biasdense_106/kerneldense_106/biasdense_107/kerneldense_107/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcount*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*.
f)R'
%__inference__traced_restore_180407549÷
é
ë
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407247

inputs,
(conv2d_21_statefulpartitionedcall_args_1,
(conv2d_21_statefulpartitionedcall_args_2,
(dense_105_statefulpartitionedcall_args_1,
(dense_105_statefulpartitionedcall_args_2,
(dense_106_statefulpartitionedcall_args_1,
(dense_106_statefulpartitionedcall_args_2,
(dense_107_statefulpartitionedcall_args_1,
(dense_107_statefulpartitionedcall_args_2
identity¢!conv2d_21/StatefulPartitionedCall¢!dense_105/StatefulPartitionedCall¢!dense_106/StatefulPartitionedCall¢!dense_107/StatefulPartitionedCall»
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_21_statefulpartitionedcall_args_1(conv2d_21_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_conv2d_21_layer_call_and_return_conditional_losses_1804070832#
!conv2d_21/StatefulPartitionedCallì
flatten_21/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_flatten_21_layer_call_and_return_conditional_losses_1804071042
flatten_21/PartitionedCallÐ
!dense_105/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0(dense_105_statefulpartitionedcall_args_1(dense_105_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_105_layer_call_and_return_conditional_losses_1804071232#
!dense_105/StatefulPartitionedCall×
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0(dense_106_statefulpartitionedcall_args_1(dense_106_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_106_layer_call_and_return_conditional_losses_1804071452#
!dense_106/StatefulPartitionedCallÖ
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0(dense_107_statefulpartitionedcall_args_1(dense_107_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_107_layer_call_and_return_conditional_losses_1804071672#
!dense_107/StatefulPartitionedCall
IdentityIdentity*dense_107/StatefulPartitionedCall:output:0"^conv2d_21/StatefulPartitionedCall"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
À


1__inference_sequential_35_layer_call_fn_180407366

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_sequential_35_layer_call_and_return_conditional_losses_1804072472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ï
á
H__inference_dense_107_layer_call_and_return_conditional_losses_180407167

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs

e
I__inference_flatten_21_layer_call_and_return_conditional_losses_180407104

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
è
J
.__inference_flatten_21_layer_call_fn_180407377

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_flatten_21_layer_call_and_return_conditional_losses_1804071042
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
ý
®
-__inference_dense_105_layer_call_fn_180407395

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_105_layer_call_and_return_conditional_losses_1804071232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
û
®
-__inference_dense_107_layer_call_fn_180407429

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_107_layer_call_and_return_conditional_losses_1804071672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

ô
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407180
conv2d_21_input,
(conv2d_21_statefulpartitionedcall_args_1,
(conv2d_21_statefulpartitionedcall_args_2,
(dense_105_statefulpartitionedcall_args_1,
(dense_105_statefulpartitionedcall_args_2,
(dense_106_statefulpartitionedcall_args_1,
(dense_106_statefulpartitionedcall_args_2,
(dense_107_statefulpartitionedcall_args_1,
(dense_107_statefulpartitionedcall_args_2
identity¢!conv2d_21/StatefulPartitionedCall¢!dense_105/StatefulPartitionedCall¢!dense_106/StatefulPartitionedCall¢!dense_107/StatefulPartitionedCallÄ
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallconv2d_21_input(conv2d_21_statefulpartitionedcall_args_1(conv2d_21_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_conv2d_21_layer_call_and_return_conditional_losses_1804070832#
!conv2d_21/StatefulPartitionedCallì
flatten_21/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_flatten_21_layer_call_and_return_conditional_losses_1804071042
flatten_21/PartitionedCallÐ
!dense_105/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0(dense_105_statefulpartitionedcall_args_1(dense_105_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_105_layer_call_and_return_conditional_losses_1804071232#
!dense_105/StatefulPartitionedCall×
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0(dense_106_statefulpartitionedcall_args_1(dense_106_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_106_layer_call_and_return_conditional_losses_1804071452#
!dense_106/StatefulPartitionedCallÖ
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0(dense_107_statefulpartitionedcall_args_1(dense_107_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_107_layer_call_and_return_conditional_losses_1804071672#
!dense_107/StatefulPartitionedCall
IdentityIdentity*dense_107/StatefulPartitionedCall:output:0"^conv2d_21/StatefulPartitionedCall"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_21_input
º'
ó
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407309

inputs,
(conv2d_21_conv2d_readvariableop_resource-
)conv2d_21_biasadd_readvariableop_resource,
(dense_105_matmul_readvariableop_resource-
)dense_105_biasadd_readvariableop_resource,
(dense_106_matmul_readvariableop_resource-
)dense_106_biasadd_readvariableop_resource,
(dense_107_matmul_readvariableop_resource-
)dense_107_biasadd_readvariableop_resource
identity¢ conv2d_21/BiasAdd/ReadVariableOp¢conv2d_21/Conv2D/ReadVariableOp¢ dense_105/BiasAdd/ReadVariableOp¢dense_105/MatMul/ReadVariableOp¢ dense_106/BiasAdd/ReadVariableOp¢dense_106/MatMul/ReadVariableOp¢ dense_107/BiasAdd/ReadVariableOp¢dense_107/MatMul/ReadVariableOp´
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_21/Conv2D/ReadVariableOpÃ
conv2d_21/Conv2DConv2Dinputs'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_21/Conv2D«
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp±
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_21/BiasAddu
flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_21/Const
flatten_21/ReshapeReshapeconv2d_21/BiasAdd:output:0flatten_21/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
flatten_21/Reshape­
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource* 
_output_shapes
:
<*
dtype02!
dense_105/MatMul/ReadVariableOp§
dense_105/MatMulMatMulflatten_21/Reshape:output:0'dense_105/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_105/MatMul«
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_105/BiasAdd/ReadVariableOpª
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_105/BiasAddw
dense_105/TanhTanhdense_105/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_105/Tanh­
dense_106/MatMul/ReadVariableOpReadVariableOp(dense_106_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_106/MatMul/ReadVariableOp
dense_106/MatMulMatMuldense_105/Tanh:y:0'dense_106/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_106/MatMul«
 dense_106/BiasAdd/ReadVariableOpReadVariableOp)dense_106_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_106/BiasAdd/ReadVariableOpª
dense_106/BiasAddBiasAdddense_106/MatMul:product:0(dense_106/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_106/BiasAdd¬
dense_107/MatMul/ReadVariableOpReadVariableOp(dense_107_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_107/MatMul/ReadVariableOp¥
dense_107/MatMulMatMuldense_106/BiasAdd:output:0'dense_107/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_107/MatMulª
 dense_107/BiasAdd/ReadVariableOpReadVariableOp)dense_107_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_107/BiasAdd/ReadVariableOp©
dense_107/BiasAddBiasAdddense_107/MatMul:product:0(dense_107/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_107/BiasAdd
IdentityIdentitydense_107/BiasAdd:output:0!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp!^dense_106/BiasAdd/ReadVariableOp ^dense_106/MatMul/ReadVariableOp!^dense_107/BiasAdd/ReadVariableOp ^dense_107/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2D
 dense_106/BiasAdd/ReadVariableOp dense_106/BiasAdd/ReadVariableOp2B
dense_106/MatMul/ReadVariableOpdense_106/MatMul/ReadVariableOp2D
 dense_107/BiasAdd/ReadVariableOp dense_107/BiasAdd/ReadVariableOp2B
dense_107/MatMul/ReadVariableOpdense_107/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ý
®
-__inference_dense_106_layer_call_fn_180407412

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_106_layer_call_and_return_conditional_losses_1804071452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ï
á
H__inference_dense_107_layer_call_and_return_conditional_losses_180407422

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs

ô
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407197
conv2d_21_input,
(conv2d_21_statefulpartitionedcall_args_1,
(conv2d_21_statefulpartitionedcall_args_2,
(dense_105_statefulpartitionedcall_args_1,
(dense_105_statefulpartitionedcall_args_2,
(dense_106_statefulpartitionedcall_args_1,
(dense_106_statefulpartitionedcall_args_2,
(dense_107_statefulpartitionedcall_args_1,
(dense_107_statefulpartitionedcall_args_2
identity¢!conv2d_21/StatefulPartitionedCall¢!dense_105/StatefulPartitionedCall¢!dense_106/StatefulPartitionedCall¢!dense_107/StatefulPartitionedCallÄ
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallconv2d_21_input(conv2d_21_statefulpartitionedcall_args_1(conv2d_21_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_conv2d_21_layer_call_and_return_conditional_losses_1804070832#
!conv2d_21/StatefulPartitionedCallì
flatten_21/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_flatten_21_layer_call_and_return_conditional_losses_1804071042
flatten_21/PartitionedCallÐ
!dense_105/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0(dense_105_statefulpartitionedcall_args_1(dense_105_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_105_layer_call_and_return_conditional_losses_1804071232#
!dense_105/StatefulPartitionedCall×
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0(dense_106_statefulpartitionedcall_args_1(dense_106_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_106_layer_call_and_return_conditional_losses_1804071452#
!dense_106/StatefulPartitionedCallÖ
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0(dense_107_statefulpartitionedcall_args_1(dense_107_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_107_layer_call_and_return_conditional_losses_1804071672#
!dense_107/StatefulPartitionedCall
IdentityIdentity*dense_107/StatefulPartitionedCall:output:0"^conv2d_21/StatefulPartitionedCall"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_21_input
©


'__inference_signature_wrapper_180407278
conv2d_21_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallconv2d_21_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference__wrapped_model_1804070712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_21_input
Ç	
á
H__inference_dense_105_layer_call_and_return_conditional_losses_180407388

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
<*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Û


1__inference_sequential_35_layer_call_fn_180407228
conv2d_21_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallconv2d_21_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_sequential_35_layer_call_and_return_conditional_losses_1804072172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_21_input
À


1__inference_sequential_35_layer_call_fn_180407353

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_sequential_35_layer_call_and_return_conditional_losses_1804072172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ø=

%__inference__traced_restore_180407549
file_prefix%
!assignvariableop_conv2d_21_kernel%
!assignvariableop_1_conv2d_21_bias'
#assignvariableop_2_dense_105_kernel%
!assignvariableop_3_dense_105_bias'
#assignvariableop_4_dense_106_kernel%
!assignvariableop_5_dense_106_bias'
#assignvariableop_6_dense_107_kernel%
!assignvariableop_7_dense_107_bias
assignvariableop_8_sgd_iter 
assignvariableop_9_sgd_decay)
%assignvariableop_10_sgd_learning_rate$
 assignvariableop_11_sgd_momentum
assignvariableop_12_total
assignvariableop_13_count
identity_15¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1ç
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ó
valueéBæB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesª
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesñ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_21_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_21_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_105_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_105_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_106_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_106_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_107_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_107_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_sgd_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOp%assignvariableop_10_sgd_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOp assignvariableop_11_sgd_momentumIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
ô
á
H__inference_dense_106_layer_call_and_return_conditional_losses_180407405

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
§'

"__inference__traced_save_180407495
file_prefix/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_dense_105_kernel_read_readvariableop-
)savev2_dense_105_bias_read_readvariableop/
+savev2_dense_106_kernel_read_readvariableop-
)savev2_dense_106_bias_read_readvariableop/
+savev2_dense_107_kernel_read_readvariableop-
)savev2_dense_107_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1¥
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a6d7a6bd8c0549149736701c8d3c4365/part2
StringJoin/inputs_1

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameá
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ó
valueéBæB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names¤
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÿ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_dense_105_kernel_read_readvariableop)savev2_dense_105_bias_read_readvariableop+savev2_dense_106_kernel_read_readvariableop)savev2_dense_106_bias_read_readvariableop+savev2_dense_107_kernel_read_readvariableop)savev2_dense_107_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*t
_input_shapesc
a: :::
<::
::	:: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
Ç	
á
H__inference_dense_105_layer_call_and_return_conditional_losses_180407123

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
<*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
é
ë
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407217

inputs,
(conv2d_21_statefulpartitionedcall_args_1,
(conv2d_21_statefulpartitionedcall_args_2,
(dense_105_statefulpartitionedcall_args_1,
(dense_105_statefulpartitionedcall_args_2,
(dense_106_statefulpartitionedcall_args_1,
(dense_106_statefulpartitionedcall_args_2,
(dense_107_statefulpartitionedcall_args_1,
(dense_107_statefulpartitionedcall_args_2
identity¢!conv2d_21/StatefulPartitionedCall¢!dense_105/StatefulPartitionedCall¢!dense_106/StatefulPartitionedCall¢!dense_107/StatefulPartitionedCall»
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_21_statefulpartitionedcall_args_1(conv2d_21_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_conv2d_21_layer_call_and_return_conditional_losses_1804070832#
!conv2d_21/StatefulPartitionedCallì
flatten_21/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_flatten_21_layer_call_and_return_conditional_losses_1804071042
flatten_21/PartitionedCallÐ
!dense_105/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0(dense_105_statefulpartitionedcall_args_1(dense_105_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_105_layer_call_and_return_conditional_losses_1804071232#
!dense_105/StatefulPartitionedCall×
!dense_106/StatefulPartitionedCallStatefulPartitionedCall*dense_105/StatefulPartitionedCall:output:0(dense_106_statefulpartitionedcall_args_1(dense_106_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_106_layer_call_and_return_conditional_losses_1804071452#
!dense_106/StatefulPartitionedCallÖ
!dense_107/StatefulPartitionedCallStatefulPartitionedCall*dense_106/StatefulPartitionedCall:output:0(dense_107_statefulpartitionedcall_args_1(dense_107_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_dense_107_layer_call_and_return_conditional_losses_1804071672#
!dense_107/StatefulPartitionedCall
IdentityIdentity*dense_107/StatefulPartitionedCall:output:0"^conv2d_21/StatefulPartitionedCall"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall"^dense_107/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall2F
!dense_107/StatefulPartitionedCall!dense_107/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ü

á
H__inference_conv2d_21_layer_call_and_return_conditional_losses_180407083

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp·
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd°
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
í2
´
$__inference__wrapped_model_180407071
conv2d_21_input:
6sequential_35_conv2d_21_conv2d_readvariableop_resource;
7sequential_35_conv2d_21_biasadd_readvariableop_resource:
6sequential_35_dense_105_matmul_readvariableop_resource;
7sequential_35_dense_105_biasadd_readvariableop_resource:
6sequential_35_dense_106_matmul_readvariableop_resource;
7sequential_35_dense_106_biasadd_readvariableop_resource:
6sequential_35_dense_107_matmul_readvariableop_resource;
7sequential_35_dense_107_biasadd_readvariableop_resource
identity¢.sequential_35/conv2d_21/BiasAdd/ReadVariableOp¢-sequential_35/conv2d_21/Conv2D/ReadVariableOp¢.sequential_35/dense_105/BiasAdd/ReadVariableOp¢-sequential_35/dense_105/MatMul/ReadVariableOp¢.sequential_35/dense_106/BiasAdd/ReadVariableOp¢-sequential_35/dense_106/MatMul/ReadVariableOp¢.sequential_35/dense_107/BiasAdd/ReadVariableOp¢-sequential_35/dense_107/MatMul/ReadVariableOpÞ
-sequential_35/conv2d_21/Conv2D/ReadVariableOpReadVariableOp6sequential_35_conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02/
-sequential_35/conv2d_21/Conv2D/ReadVariableOpö
sequential_35/conv2d_21/Conv2DConv2Dconv2d_21_input5sequential_35/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2 
sequential_35/conv2d_21/Conv2DÕ
.sequential_35/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp7sequential_35_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.sequential_35/conv2d_21/BiasAdd/ReadVariableOpé
sequential_35/conv2d_21/BiasAddBiasAdd'sequential_35/conv2d_21/Conv2D:output:06sequential_35/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_35/conv2d_21/BiasAdd
sequential_35/flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2 
sequential_35/flatten_21/ConstÕ
 sequential_35/flatten_21/ReshapeReshape(sequential_35/conv2d_21/BiasAdd:output:0'sequential_35/flatten_21/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2"
 sequential_35/flatten_21/Reshape×
-sequential_35/dense_105/MatMul/ReadVariableOpReadVariableOp6sequential_35_dense_105_matmul_readvariableop_resource* 
_output_shapes
:
<*
dtype02/
-sequential_35/dense_105/MatMul/ReadVariableOpß
sequential_35/dense_105/MatMulMatMul)sequential_35/flatten_21/Reshape:output:05sequential_35/dense_105/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_35/dense_105/MatMulÕ
.sequential_35/dense_105/BiasAdd/ReadVariableOpReadVariableOp7sequential_35_dense_105_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.sequential_35/dense_105/BiasAdd/ReadVariableOpâ
sequential_35/dense_105/BiasAddBiasAdd(sequential_35/dense_105/MatMul:product:06sequential_35/dense_105/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_35/dense_105/BiasAdd¡
sequential_35/dense_105/TanhTanh(sequential_35/dense_105/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_35/dense_105/Tanh×
-sequential_35/dense_106/MatMul/ReadVariableOpReadVariableOp6sequential_35_dense_106_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-sequential_35/dense_106/MatMul/ReadVariableOpÖ
sequential_35/dense_106/MatMulMatMul sequential_35/dense_105/Tanh:y:05sequential_35/dense_106/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_35/dense_106/MatMulÕ
.sequential_35/dense_106/BiasAdd/ReadVariableOpReadVariableOp7sequential_35_dense_106_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.sequential_35/dense_106/BiasAdd/ReadVariableOpâ
sequential_35/dense_106/BiasAddBiasAdd(sequential_35/dense_106/MatMul:product:06sequential_35/dense_106/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_35/dense_106/BiasAddÖ
-sequential_35/dense_107/MatMul/ReadVariableOpReadVariableOp6sequential_35_dense_107_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-sequential_35/dense_107/MatMul/ReadVariableOpÝ
sequential_35/dense_107/MatMulMatMul(sequential_35/dense_106/BiasAdd:output:05sequential_35/dense_107/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_35/dense_107/MatMulÔ
.sequential_35/dense_107/BiasAdd/ReadVariableOpReadVariableOp7sequential_35_dense_107_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_35/dense_107/BiasAdd/ReadVariableOpá
sequential_35/dense_107/BiasAddBiasAdd(sequential_35/dense_107/MatMul:product:06sequential_35/dense_107/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_35/dense_107/BiasAdd
IdentityIdentity(sequential_35/dense_107/BiasAdd:output:0/^sequential_35/conv2d_21/BiasAdd/ReadVariableOp.^sequential_35/conv2d_21/Conv2D/ReadVariableOp/^sequential_35/dense_105/BiasAdd/ReadVariableOp.^sequential_35/dense_105/MatMul/ReadVariableOp/^sequential_35/dense_106/BiasAdd/ReadVariableOp.^sequential_35/dense_106/MatMul/ReadVariableOp/^sequential_35/dense_107/BiasAdd/ReadVariableOp.^sequential_35/dense_107/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2`
.sequential_35/conv2d_21/BiasAdd/ReadVariableOp.sequential_35/conv2d_21/BiasAdd/ReadVariableOp2^
-sequential_35/conv2d_21/Conv2D/ReadVariableOp-sequential_35/conv2d_21/Conv2D/ReadVariableOp2`
.sequential_35/dense_105/BiasAdd/ReadVariableOp.sequential_35/dense_105/BiasAdd/ReadVariableOp2^
-sequential_35/dense_105/MatMul/ReadVariableOp-sequential_35/dense_105/MatMul/ReadVariableOp2`
.sequential_35/dense_106/BiasAdd/ReadVariableOp.sequential_35/dense_106/BiasAdd/ReadVariableOp2^
-sequential_35/dense_106/MatMul/ReadVariableOp-sequential_35/dense_106/MatMul/ReadVariableOp2`
.sequential_35/dense_107/BiasAdd/ReadVariableOp.sequential_35/dense_107/BiasAdd/ReadVariableOp2^
-sequential_35/dense_107/MatMul/ReadVariableOp-sequential_35/dense_107/MatMul/ReadVariableOp:/ +
)
_user_specified_nameconv2d_21_input

e
I__inference_flatten_21_layer_call_and_return_conditional_losses_180407372

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
º'
ó
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407340

inputs,
(conv2d_21_conv2d_readvariableop_resource-
)conv2d_21_biasadd_readvariableop_resource,
(dense_105_matmul_readvariableop_resource-
)dense_105_biasadd_readvariableop_resource,
(dense_106_matmul_readvariableop_resource-
)dense_106_biasadd_readvariableop_resource,
(dense_107_matmul_readvariableop_resource-
)dense_107_biasadd_readvariableop_resource
identity¢ conv2d_21/BiasAdd/ReadVariableOp¢conv2d_21/Conv2D/ReadVariableOp¢ dense_105/BiasAdd/ReadVariableOp¢dense_105/MatMul/ReadVariableOp¢ dense_106/BiasAdd/ReadVariableOp¢dense_106/MatMul/ReadVariableOp¢ dense_107/BiasAdd/ReadVariableOp¢dense_107/MatMul/ReadVariableOp´
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_21/Conv2D/ReadVariableOpÃ
conv2d_21/Conv2DConv2Dinputs'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_21/Conv2D«
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp±
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_21/BiasAddu
flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_21/Const
flatten_21/ReshapeReshapeconv2d_21/BiasAdd:output:0flatten_21/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
flatten_21/Reshape­
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource* 
_output_shapes
:
<*
dtype02!
dense_105/MatMul/ReadVariableOp§
dense_105/MatMulMatMulflatten_21/Reshape:output:0'dense_105/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_105/MatMul«
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_105/BiasAdd/ReadVariableOpª
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_105/BiasAddw
dense_105/TanhTanhdense_105/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_105/Tanh­
dense_106/MatMul/ReadVariableOpReadVariableOp(dense_106_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_106/MatMul/ReadVariableOp
dense_106/MatMulMatMuldense_105/Tanh:y:0'dense_106/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_106/MatMul«
 dense_106/BiasAdd/ReadVariableOpReadVariableOp)dense_106_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_106/BiasAdd/ReadVariableOpª
dense_106/BiasAddBiasAdddense_106/MatMul:product:0(dense_106/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_106/BiasAdd¬
dense_107/MatMul/ReadVariableOpReadVariableOp(dense_107_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_107/MatMul/ReadVariableOp¥
dense_107/MatMulMatMuldense_106/BiasAdd:output:0'dense_107/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_107/MatMulª
 dense_107/BiasAdd/ReadVariableOpReadVariableOp)dense_107_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_107/BiasAdd/ReadVariableOp©
dense_107/BiasAddBiasAdddense_107/MatMul:product:0(dense_107/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_107/BiasAdd
IdentityIdentitydense_107/BiasAdd:output:0!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp!^dense_106/BiasAdd/ReadVariableOp ^dense_106/MatMul/ReadVariableOp!^dense_107/BiasAdd/ReadVariableOp ^dense_107/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2D
 dense_106/BiasAdd/ReadVariableOp dense_106/BiasAdd/ReadVariableOp2B
dense_106/MatMul/ReadVariableOpdense_106/MatMul/ReadVariableOp2D
 dense_107/BiasAdd/ReadVariableOp dense_107/BiasAdd/ReadVariableOp2B
dense_107/MatMul/ReadVariableOpdense_107/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Û


1__inference_sequential_35_layer_call_fn_180407258
conv2d_21_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallconv2d_21_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_sequential_35_layer_call_and_return_conditional_losses_1804072472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_21_input
Ê
®
-__inference_conv2d_21_layer_call_fn_180407091

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_conv2d_21_layer_call_and_return_conditional_losses_1804070832
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ô
á
H__inference_dense_106_layer_call_and_return_conditional_losses_180407145

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs"¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ä
serving_default°
S
conv2d_21_input@
!serving_default_conv2d_21_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_1070
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:®²
æ+
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
Q_default_save_signature
*R&call_and_return_all_conditional_losses
S__call__"å(
_tf_keras_sequentialÆ({"class_name": "Sequential", "name": "sequential_35", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_35", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [2, 2], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_105", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_106", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_107", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_35", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [2, 2], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_105", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_106", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_107", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": [{"class_name": "MeanSquaredError", "config": {"name": "mean_squared_error", "dtype": "float32"}}], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.004999999888241291, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
¹"¶
_tf_keras_input_layer{"class_name": "InputLayer", "name": "conv2d_21_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 6, 7, 1], "config": {"batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_21_input"}}
¹

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*T&call_and_return_all_conditional_losses
U__call__"
_tf_keras_layerú{"class_name": "Conv2D", "name": "conv2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 6, 7, 1], "config": {"name": "conv2d_21", "trainable": true, "batch_input_shape": [null, 6, 7, 1], "dtype": "float32", "filters": 256, "kernel_size": [2, 2], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
²
regularization_losses
	variables
trainable_variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"£
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"ê
_tf_keras_layerÐ{"class_name": "Dense", "name": "dense_105", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_105", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7680}}}}


kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_106", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_106", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
*\&call_and_return_all_conditional_losses
]__call__"é
_tf_keras_layerÏ{"class_name": "Dense", "name": "dense_107", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_107", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0, "stddev": 0.1, "seed": 10}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
I
)iter
	*decay
+learning_rate
,momentum"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
·
-metrics
.non_trainable_variables
regularization_losses
		variables

trainable_variables
/layer_regularization_losses

0layers
S__call__
Q_default_save_signature
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
,
^serving_default"
signature_map
+:)2conv2d_21/kernel
:2conv2d_21/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

1metrics
2non_trainable_variables
regularization_losses
	variables
trainable_variables
3layer_regularization_losses

4layers
U__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

5metrics
6non_trainable_variables
regularization_losses
	variables
trainable_variables
7layer_regularization_losses

8layers
W__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
$:"
<2dense_105/kernel
:2dense_105/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

9metrics
:non_trainable_variables
regularization_losses
	variables
trainable_variables
;layer_regularization_losses

<layers
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
$:"
2dense_106/kernel
:2dense_106/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

=metrics
>non_trainable_variables
regularization_losses
 	variables
!trainable_variables
?layer_regularization_losses

@layers
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
#:!	2dense_107/kernel
:2dense_107/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper

Ametrics
Bnon_trainable_variables
%regularization_losses
&	variables
'trainable_variables
Clayer_regularization_losses

Dlayers
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
	Ftotal
	Gcount
H
_fn_kwargs
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
*_&call_and_return_all_conditional_losses
`__call__"ø
_tf_keras_layerÞ{"class_name": "MeanSquaredError", "name": "mean_squared_error", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mean_squared_error", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper

Mmetrics
Nnon_trainable_variables
Iregularization_losses
J	variables
Ktrainable_variables
Olayer_regularization_losses

Players
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ò2ï
$__inference__wrapped_model_180407071Æ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *6¢3
1.
conv2d_21_inputÿÿÿÿÿÿÿÿÿ
þ2û
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407340
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407309
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407180
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407197À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
1__inference_sequential_35_layer_call_fn_180407258
1__inference_sequential_35_layer_call_fn_180407366
1__inference_sequential_35_layer_call_fn_180407228
1__inference_sequential_35_layer_call_fn_180407353À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
§2¤
H__inference_conv2d_21_layer_call_and_return_conditional_losses_180407083×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
-__inference_conv2d_21_layer_call_fn_180407091×
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ó2ð
I__inference_flatten_21_layer_call_and_return_conditional_losses_180407372¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_flatten_21_layer_call_fn_180407377¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_dense_105_layer_call_and_return_conditional_losses_180407388¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_dense_105_layer_call_fn_180407395¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_dense_106_layer_call_and_return_conditional_losses_180407405¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_dense_106_layer_call_fn_180407412¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_dense_107_layer_call_and_return_conditional_losses_180407422¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_dense_107_layer_call_fn_180407429¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
>B<
'__inference_signature_wrapper_180407278conv2d_21_input
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 ¬
$__inference__wrapped_model_180407071#$@¢=
6¢3
1.
conv2d_21_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_107# 
	dense_107ÿÿÿÿÿÿÿÿÿÞ
H__inference_conv2d_21_layer_call_and_return_conditional_losses_180407083I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
-__inference_conv2d_21_layer_call_fn_180407091I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
H__inference_dense_105_layer_call_and_return_conditional_losses_180407388^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ<
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_105_layer_call_fn_180407395Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ<
ª "ÿÿÿÿÿÿÿÿÿª
H__inference_dense_106_layer_call_and_return_conditional_losses_180407405^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_106_layer_call_fn_180407412Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
H__inference_dense_107_layer_call_and_return_conditional_losses_180407422]#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_107_layer_call_fn_180407429P#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¯
I__inference_flatten_21_layer_call_and_return_conditional_losses_180407372b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ<
 
.__inference_flatten_21_layer_call_fn_180407377U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ<Ë
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407180{#$H¢E
>¢;
1.
conv2d_21_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407197{#$H¢E
>¢;
1.
conv2d_21_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407309r#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
L__inference_sequential_35_layer_call_and_return_conditional_losses_180407340r#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
1__inference_sequential_35_layer_call_fn_180407228n#$H¢E
>¢;
1.
conv2d_21_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ£
1__inference_sequential_35_layer_call_fn_180407258n#$H¢E
>¢;
1.
conv2d_21_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
1__inference_sequential_35_layer_call_fn_180407353e#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
1__inference_sequential_35_layer_call_fn_180407366e#$?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÂ
'__inference_signature_wrapper_180407278#$S¢P
¢ 
IªF
D
conv2d_21_input1.
conv2d_21_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_107# 
	dense_107ÿÿÿÿÿÿÿÿÿ