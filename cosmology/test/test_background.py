def test_default_implementations():

    from cosmology.background import Cosmology

    for method, requires, default in Cosmology._default_methods():
        subclass = type('TestCosmology', (Cosmology,),
                        {name: lambda x: x for name in requires})
        assert getattr(subclass, method) is default
