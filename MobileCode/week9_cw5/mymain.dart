import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

main() {
  runApp(const ShowInf());
}

class ShowInf extends StatefulWidget {
  const ShowInf({super.key});

  @override
  State<ShowInf> createState() => _ShowInfState();
}

class _ShowInfState extends State<ShowInf> {
  List list = [];
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _addressController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _phoneController = TextEditingController();

  Future<String> listData() async {
    var response = await http.get(Uri.http('10.0.2.2:8080', 'emp'),
        headers: {"Accept": "application/json"});
    print("Response status: ${response.statusCode}");
    print("Response body: ${response.body}");
    setState(() {
      list = jsonDecode(response.body);
    });
    return 'Success';
  }

  @override
  void initState() {
    super.initState();
    listData();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('DB Test'),
      ),
      body: Center(
        child: ListView.builder(
          itemCount: list.length,
          itemBuilder: (BuildContext context, int index) {
            return Card(
              child: ListTile(
                title: Row(
                  children: [
                    Expanded(child: Text(list[index]['name'])),
                    Expanded(child: Text(list[index]['email'])),
                  ],
                ),
                leading: Text(list[index]['id'].toString()),
                trailing: Wrap(
                  spacing: 5,
                  children: [
                    IconButton(
                      onPressed: () {
                        Map data = {
                          'id': list[index]['id'],
                          'name': list[index]['name'],
                          'address': list[index]['address'],
                          'email': list[index]['email'],
                          'phone': list[index]['phone']
                        };
                        _showedit(data);
                      },
                      icon: Icon(Icons.edit),
                      color: Colors.green,
                    ),
                    IconButton(
                        onPressed: () => _showDel(list[index]["id"]),
                        icon: const Icon(
                          Icons.delete_outline,
                          color: Colors.red,
                        )),
                  ],
                ),
              ),
            );
          },
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          _addNewDialog();
        },
        child: const Icon(Icons.add),
      ),
    );
  }

  Future<void> _addNewDialog() async {
    return showDialog(
        barrierDismissible: false,
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text('Add emp'),
            content: SingleChildScrollView(
                child: ListBody(
              children: <Widget>[
                TextField(
                  controller: _nameController,
                  decoration: const InputDecoration(
                      labelText: 'Name', hintText: "Enter emp Name"),
                ),
                TextField(
                  controller: _emailController,
                  decoration: const InputDecoration(
                      labelText: 'Email', hintText: "Enter emp Email"),
                ),
                TextField(
                  controller: _phoneController,
                  decoration: const InputDecoration(
                      labelText: 'Phone', hintText: "Enter emp Phone"),
                ),
                TextField(
                  controller: _addressController,
                  decoration: const InputDecoration(
                      labelText: 'Address', hintText: "Enter emp Address"),
                ),
                const Text('กรอกข้อมูลให้เรียบร้อยแล้วกด ยืนยัน'),
              ],
            )),
            actions: <Widget>[
              TextButton(
                  onPressed: () {
                    add_data();
                    Navigator.of(context).pop();
                  },
                  child: Text('ยืนยัน')),
            ],
          );
        });
  }

  Future<void> _showDel(int id) async {
    return showDialog<void>(
        barrierDismissible: false,
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('ลบข้อมูล ${id}'),
            content: SingleChildScrollView(
                child: ListBody(
              children: <Widget>[
                Text('ยืนยันการลบข้อมูล กด ยืนยัน'),
              ],
            )),
            actions: <Widget>[
              TextButton(
                  onPressed: () {
                    del_data(id);
                    Navigator.of(context).pop();
                  },
                  child: Text('ยืนยัน')),
            ],
          );
        });
  }

  void add_data() async {
    Map data = {
      'name': _nameController.text,
      'email': _emailController.text,
      'phone': _phoneController.text,
      'address': _addressController.text
    };
    var body = jsonEncode(data);
    var response = await http.post(Uri.http('10.0.2.2:8080', 'create'),
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: body);
    print("Response status: ${response.statusCode}");
    print("Response body: ${response.body}");
    listData();
  }

  void del_data(int id) async {
    var response = await http.delete(Uri.http('10.0.2.2:8080', 'delete/$id'),
        headers: {
          "Content-Type": "application/json; charset=UTF-8",
          "Accept": "application/json"
        });
    print("Response status: ${response.statusCode}");
    print("Response body: ${response.body}");
    listData();
  }

  Future<void> _showedit(Map data) async {
    _nameController.text = data['name'];
    _emailController.text = data['email'];
    _phoneController.text = data['phone'];
    _addressController.text = data['address'];
    return showDialog<void>(
        barrierDismissible: false,
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text('ทดสอบการ Edit'),
            content: SingleChildScrollView(
                child: ListBody(
              children: <Widget>[
                TextField(
                  controller: _nameController,
                  decoration: const InputDecoration(
                      labelText: 'Name', hintText: "Enter emp Name"),
                ),
                TextField(
                  controller: _emailController,
                  decoration: const InputDecoration(
                      labelText: 'Email', hintText: "Enter emp Email"),
                ),
                TextField(
                  controller: _phoneController,
                  decoration: const InputDecoration(
                      labelText: 'Phone', hintText: "Enter emp Phone"),
                ),
                TextField(
                  controller: _addressController,
                  decoration: const InputDecoration(
                      labelText: 'Address', hintText: "Enter emp Address"),
                ),
                const Text('ปรับปรุงข้อมูลให้เรียบร้อยแล้วกด ยืนยัน'),
              ],
            )),
            actions: <Widget>[
              TextButton(
                  onPressed: () {
                    edit_data(data['id']);
                    Navigator.of(context).pop();
                  },
                  child: Text('ยืนยัน')),
            ],
          );
        });
  }

  void edit_data(id) async {
    Map data = {
      'id': id,
      'name': _nameController.text,
      'email': _emailController.text,
      'phone': _phoneController.text,
      'address': _addressController.text
    };
    var body = jsonEncode(data);
    var response = await http.put(Uri.http('10.0.2.2:8080', 'update/$id'),
        headers: {
          "Content-Type": "application/json; charset=UTF-8",
        },
        body: body);
    print("Response status: ${response.statusCode}");
    print("Response body: ${response.body}");
    listData();
  }
}
