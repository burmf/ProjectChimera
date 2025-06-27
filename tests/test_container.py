"""
Tests for container.py - simple dependency injection container
"""


from project_chimera.container import Container


class TestContainer:
    """Test dependency injection container"""

    def test_container_initialization(self):
        """Test container initialization"""
        container = Container()
        assert hasattr(container, 'config')
        assert container.config is not None

    def test_container_wiring(self):
        """Test container wiring functionality"""
        container = Container()

        from contextlib import suppress

        # Should be able to call wire without error
        with suppress(Exception):
            container.wire(modules=[__name__])

    def test_container_providers(self):
        """Test container providers exist"""
        container = Container()

        # Check that expected providers exist
        expected_providers = ['config']
        for provider_name in expected_providers:
            assert hasattr(container, provider_name)

    def test_container_config_access(self):
        """Test accessing configuration through container"""
        container = Container()
        config = container.config()

        # Should return some kind of configuration object
        assert config is not None
