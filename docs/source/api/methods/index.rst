Methods
=======

This section provides documentation for the different confidence interval calculation methods implemented in ExactCIs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   unconditional
   conditional
   blaker
   midp
   wald
   clopper_pearson

Unconditional Method
-------------------

The unconditional method (Barnard's test) provides exact confidence intervals without conditioning on the margins.

.. automodule:: exactcis.methods.unconditional
   :members:
   :undoc-members:
   :show-inheritance:

Conditional Method
----------------

The conditional method (Fisher's exact test) provides confidence intervals conditioning on the margins.

.. automodule:: exactcis.methods.conditional
   :members:
   :undoc-members:
   :show-inheritance:

Blaker's Method
-------------

Blaker's method provides an alternative approach to calculating exact confidence intervals.

.. automodule:: exactcis.methods.blaker
   :members:
   :undoc-members:
   :show-inheritance:

Mid-P Method
----------

The Mid-P method provides a less conservative alternative to Fisher's exact test by using the mid-p-value.

.. automodule:: exactcis.methods.midp
   :members:
   :undoc-members:
   :show-inheritance:

Wald Method
---------

The Wald method provides asymptotic confidence intervals based on the normal approximation.

.. automodule:: exactcis.methods.wald
   :members:
   :undoc-members:
   :show-inheritance:

Clopper-Pearson Method
-------------------

The Clopper-Pearson method provides exact confidence intervals for binomial proportions using the binomial cumulative distribution function.

.. automodule:: exactcis.methods.clopper_pearson
   :members:
   :undoc-members:
   :show-inheritance: