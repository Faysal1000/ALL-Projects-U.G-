namespace LifeinnovirorMentalHealthConsultency.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class addingPassFieldToPatient : DbMigration
    {
        public override void Up()
        {
            AddColumn("dbo.Patients", "PasswordHash", c => c.String(maxLength: 255));
        }
        
        public override void Down()
        {
            DropColumn("dbo.Patients", "PasswordHash");
        }
    }
}
