import 'package:flutter/material.dart';

class formShopping extends StatelessWidget {
  formShopping({Key? key, required this.productName, required this.customerName}) : super(key: key);
  String productName;
  String customerName;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Shopping Screen'),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
      ),
      body: Container(
        padding: EdgeInsets.all(20),
        child: ListView(
          children: [
            ListTile(
              leading: Icon(Icons.account_balance_wallet_outlined),
              title: Text(productName),
              subtitle: Text(customerName),
            ),
          ],
        ),
      ),
    );
  }
}