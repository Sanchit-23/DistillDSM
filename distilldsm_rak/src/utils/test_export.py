import unittest
import os
from downloader import download_checkpoint
from exporter import Exporter
from get_config import get_config

def create_export_test():
    class ExportTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='export')
            if not os.path.exists(cls.config['checkpoint']):
                download_checkpoint()
            cls.model_path = cls.config['checkpoint']

        def test_export_onnx(self):
            self.exporter = Exporter(self.config)
            self.exporter.export_model_onnx()
            self.assertTrue(os.path.join(os.path.split(self.model_path)[
                            0], self.config.get('model_name_onnx')))

        def test_export_ir(self):
            self.exporter = Exporter(self.config)
            model_dir = os.path.split(self.config['checkpoint'])[0]
            if not os.path.exists(os.path.join(model_dir, self.config.get('model_name_onnx'))):
                self.exporter.export_model_onnx()
            self.exporter.export_model_ir()
            name_xml = self.config['model_name'] + '.xml'
            name_bin = self.config['model_name'] + '.bin'
            xml_status = os.path.exists(os.path.join(model_dir, name_xml))
            bin_status = os.path.exists(os.path.join(model_dir, name_bin))
            self.assertTrue(xml_status)
            self.assertTrue(bin_status)

        def test_config(self):
            self.config = get_config(action='export')
            self.model_path = self.config['checkpoint']
            self.input_shape = self.config['input_shape']
            self.output_dir = os.path.split(self.model_path)[0]
            self.assertTrue(self.output_dir)
            self.assertTrue(self.model_path)
            self.assertListEqual(self.input_shape, [1, 1, 145, 128, 128])
    return ExportTest


class TestInference(create_export_test()):
    'Test case'

if __name__ == '__main__':
    unittest.main()