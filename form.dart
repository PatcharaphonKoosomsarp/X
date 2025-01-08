import 'package:flutter/material.dart';
import 'Shopping.dart';

class MyForm extends StatefulWidget {
  const MyForm({Key? key}) : super(key: key);

  @override
  State<MyForm> createState() => _MyFormState();
}

class _MyFormState extends State<MyForm> {
  var _productName;
  var _customerName;

  final _productController = TextEditingController();
  final _customerController = TextEditingController();

  void initState(){
    super.initState();
    _productController.addListener(_updateText);
    _customerController.addListener(_updateText);
  }

  void _updateText(){
    setState(() {
      _productName = _productController.text;
      _customerName = _customerController.text;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Form")),
      body: Container(
        padding: EdgeInsets.all(20),
        child: ListView(
          children: [
            TextFormField(
              controller: _productController,
              decoration: InputDecoration(
                labelText: 'Product Name',
                prefixIcon: Icon(Icons.verified_outlined),
                border: OutlineInputBorder(),
              ),
            ),

            SizedBox(height: 20.0),

            TextFormField(
              controller: _customerController,
              decoration: InputDecoration(
                labelText: 'Product Name',
                prefixIcon: Icon(Icons.verified_outlined),
                border: OutlineInputBorder(),
              ),
            ),

            SizedBox(height: 20.0),
            myBtn(context),
            SizedBox(height: 20.0),
            Text("Product Name is : $_productName"),
            Text("Customer Name is : $_customerName"),
          ],
        ),
      ),
    );
  }

  Center myBtn(BuildContext context) {
    return Center(
      child: ElevatedButton(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text("Go to Shopping "),
            Icon(Icons.add_shopping_cart_outlined),
          ],
        ),
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) {
                return formShopping(productName: _productController.text,customerName: _customerController.text,);
              },
            ),
          );
        },
        style: ElevatedButton.styleFrom(
          padding: EdgeInsets.all(20.0),
          fixedSize: Size(300, 80),
          textStyle: TextStyle(fontSize: 25, fontWeight: FontWeight.bold),
          backgroundColor: Colors.blueAccent, // ใช้แทน primary
          foregroundColor: Colors.white, // ใช้แทน onPrimary
          elevation: 15,
          shadowColor: Colors.black,
          side: BorderSide(color: Colors.black87, width: 2),
          alignment: Alignment.center,
          shape: StadiumBorder(),
        ),
      ),
    );
  }
}